from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .utils import seed_worker
from .variants import VariantSpec, apply_variant

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_pairs(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[int(key)] = value
    return result


def _read_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Record:
    image_id: int
    image_path: Path
    mask_path: Path
    label: int
    group_id: int | None = None


class CubDataset(Dataset):
    def __init__(
        self,
        config: dict,
        split: str,
        need_mask: bool,
        variant: VariantSpec | None = None,
        variants: list[VariantSpec] | None = None,
    ) -> None:
        data = config["data"]
        self.split = split
        self.training = split == "train"
        self.need_mask = need_mask or variant is not None
        self.variant = variant
        self.variants = variants
        self.global_seed = int(config["seed"])
        self.image_size = int(data["image_size"])
        self.resize_size = int(data["resize_size"])
        cub_root = Path(data["cub_root"])
        mask_root = Path(data["segmentation_root"])
        split_dir = Path(data["split_dir"])
        paths = _read_pairs(cub_root / "images.txt")
        labels = {key: int(value) - 1 for key, value in _read_pairs(cub_root / "image_class_labels.txt").items()}
        self.records: list[Record] = []
        for image_id in _read_ids(split_dir / f"{split}.txt"):
            relative = Path(paths[image_id])
            record = Record(
                image_id=image_id,
                image_path=cub_root / "images" / relative,
                mask_path=(mask_root / relative).with_suffix(".png"),
                label=labels[image_id],
            )
            if not record.image_path.is_file() or (self.need_mask and not record.mask_path.is_file()):
                raise FileNotFoundError(record)
            self.records.append(record)
        self.cache_dir = None
        self._cached_images = None
        self._cached_masks = None
        if self.split == "test" and data.get("evaluation_cache"):
            candidate = Path(data["evaluation_cache"])
            if (candidate / "images.npy").is_file():
                cached_ids = np.load(candidate / "image_ids.npy", mmap_mode="r")
                expected_ids = np.asarray([record.image_id for record in self.records], dtype=np.int32)
                if not np.array_equal(cached_ids, expected_ids):
                    raise ValueError(f"Evaluation cache IDs do not match split: {candidate}")
                self.cache_dir = candidate

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_images"] = None
        state["_cached_masks"] = None
        return state

    def _open_cache(self) -> None:
        if self.cache_dir is not None and self._cached_images is None:
            self._cached_images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
            self._cached_masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _crop_params(image: Image.Image, generator: torch.Generator) -> tuple[int, int, int, int]:
        width, height = TF.get_image_size(image)
        area = height * width
        scale = (0.70, 1.0)
        ratio = (0.75, 1.3333333333)
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target = area * float(torch.empty(1).uniform_(scale[0], scale[1], generator=generator))
            aspect = float(torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1], generator=generator)))
            crop_width = round(math.sqrt(target * aspect))
            crop_height = round(math.sqrt(target / aspect))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = int(torch.randint(0, height - crop_height + 1, (1,), generator=generator))
                left = int(torch.randint(0, width - crop_width + 1, (1,), generator=generator))
                return top, left, crop_height, crop_width
        input_ratio = width / height
        if input_ratio < ratio[0]:
            crop_width, crop_height = width, round(width / ratio[0])
        elif input_ratio > ratio[1]:
            crop_height, crop_width = height, round(height * ratio[1])
        else:
            crop_width, crop_height = width, height
        return (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width

    def _geometry(
        self, image: Image.Image, mask: Image.Image | None, generator: torch.Generator,
    ) -> tuple[Image.Image, Image.Image | None]:
        if self.training:
            top, left, height, width = self._crop_params(image, generator)
            image = TF.resized_crop(
                image, top, left, height, width, [self.image_size, self.image_size],
                InterpolationMode.BILINEAR, antialias=True,
            )
            if mask is not None:
                mask = TF.resized_crop(
                    mask, top, left, height, width, [self.image_size, self.image_size],
                    InterpolationMode.NEAREST,
                )
            if float(torch.rand(1, generator=generator)) < 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)
        else:
            image = TF.resize(image, self.resize_size, InterpolationMode.BILINEAR, antialias=True)
            image = TF.center_crop(image, [self.image_size, self.image_size])
            if mask is not None:
                mask = TF.resize(mask, self.resize_size, InterpolationMode.NEAREST)
                mask = TF.center_crop(mask, [self.image_size, self.image_size])
        return image, mask

    @staticmethod
    def _mask_tensor(mask: Image.Image) -> torch.Tensor:
        """Convert an ordinary binary/grayscale foreground mask to 0/1."""
        return (TF.to_tensor(mask) >= 0.5).to(torch.float32)

    def __getitem__(self, index: int | tuple) -> dict[str, torch.Tensor | int | str]:
        epoch = 0
        selected_variant = self.variant
        if isinstance(index, tuple) and len(index) == 3 and index[0] == "variant":
            _, variant_index, index = index
            if self.variants is None:
                raise RuntimeError("Variant tuple used without a variant list")
            selected_variant = self.variants[int(variant_index)]
        elif isinstance(index, tuple):
            epoch, index = index
        record = self.records[index]
        generator = torch.Generator().manual_seed(
            (self.global_seed * 1_000_003 + int(epoch) * 11_788 + record.image_id) % (2**63 - 1)
        )
        if self.cache_dir is not None:
            self._open_cache()
            image_tensor = torch.from_numpy(np.array(self._cached_images[index], copy=True)).permute(2, 0, 1).float().div_(255)
            mask_tensor = torch.from_numpy(np.array(self._cached_masks[index], copy=True)).unsqueeze(0).float()
            mask = True
        else:
            with Image.open(record.image_path) as source:
                image = source.convert("RGB")
            mask = None
            if self.need_mask:
                with Image.open(record.mask_path) as source:
                    mask = source.convert("L")
            image, mask = self._geometry(image, mask, generator)
            image_tensor = TF.to_tensor(image)
        sample: dict[str, torch.Tensor | int] = {
            "label": record.label,
            "image_id": record.image_id,
        }
        if record.group_id is not None:
            sample["group_id"] = record.group_id
        if mask is not None:
            if self.cache_dir is None:
                mask_tensor = self._mask_tensor(mask)
            # A random crop can legitimately exclude the annotated object.
            # Keep the classification sample; FCAL objectives exclude empty or
            # otherwise degenerate masks from spatial supervision.
            if selected_variant is not None:
                image_tensor = apply_variant(
                    image_tensor, mask_tensor, record.image_id, selected_variant, self.global_seed
                )
            sample["mask"] = mask_tensor
        if selected_variant is not None:
            sample["variant_key"] = selected_variant.key
        sample["image"] = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return sample


class IsicDataset(CubDataset):
    """ISIC image/lesion-mask pairs described by a prepared split manifest."""

    def __init__(
        self,
        config: dict,
        split: str,
        need_mask: bool,
        variant: VariantSpec | None = None,
        variants: list[VariantSpec] | None = None,
    ) -> None:
        data = config["data"]
        self.split = split
        self.training = split == "train"
        self.need_mask = need_mask or variant is not None or variants is not None
        self.variant = variant
        self.variants = variants
        self.global_seed = int(config["seed"])
        self.image_size = int(data["image_size"])
        self.resize_size = int(data["resize_size"])
        image_root = Path(data["isic_root"])
        mask_root = Path(data["segmentation_root"])
        split_path = Path(data["split_dir"]) / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        self.records = []
        with split_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = Record(
                    image_id=int(row["numeric_id"]),
                    image_path=image_root / row["image_path"],
                    mask_path=mask_root / row["mask_path"],
                    label=int(row["label"]),
                )
                if not record.image_path.is_file() or (self.need_mask and not record.mask_path.is_file()):
                    raise FileNotFoundError(record)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Empty ISIC split: {split_path}")
        self.cache_dir = None
        self._cached_images = None
        self._cached_masks = None


class OxfordPetDataset(CubDataset):
    """Oxford-IIIT Pet breed classification with official trimap supervision.

    Split CSVs are prepared from the official ``trainval.txt`` and ``test.txt``
    files.  Trimap value 2 is definite background; values 1 (pet) and 3
    (ambiguous boundary/accessory) form the allowed FCAL foreground region.
    """

    def __init__(
        self,
        config: dict,
        split: str,
        need_mask: bool,
        variant: VariantSpec | None = None,
        variants: list[VariantSpec] | None = None,
    ) -> None:
        data = config["data"]
        self.split = split
        self.training = split == "train"
        self.need_mask = need_mask or variant is not None or variants is not None
        self.variant = variant
        self.variants = variants
        self.global_seed = int(config["seed"])
        self.image_size = int(data["image_size"])
        self.resize_size = int(data["resize_size"])
        root = Path(data["pet_root"])
        split_path = Path(data["split_dir"]) / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        self.records = []
        with split_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = Record(
                    image_id=int(row["numeric_id"]),
                    image_path=root / row["image_path"],
                    mask_path=root / row["mask_path"],
                    label=int(row["label"]),
                )
                if not record.image_path.is_file() or (self.need_mask and not record.mask_path.is_file()):
                    raise FileNotFoundError(record)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Empty Oxford-IIIT Pet split: {split_path}")
        self.cache_dir = None
        self._cached_images = None
        self._cached_masks = None

    @staticmethod
    def _mask_tensor(mask: Image.Image) -> torch.Tensor:
        values = TF.pil_to_tensor(mask)
        return values.ne(2).to(torch.float32)


class AcdcCardiacDataset(CubDataset):
    """Prepared ACDC ED/ES/change images with patient-grouped manifests."""

    def __init__(
        self,
        config: dict,
        split: str,
        need_mask: bool,
        variant: VariantSpec | None = None,
        variants: list[VariantSpec] | None = None,
    ) -> None:
        data = config["data"]
        self.split = split
        self.training = split == "train"
        self.need_mask = need_mask or variant is not None or variants is not None
        self.variant = variant
        self.variants = variants
        self.global_seed = int(config["seed"])
        self.image_size = int(data["image_size"])
        self.resize_size = int(data.get("resize_size", self.image_size))
        root = Path(data["cardiac_root"])
        split_path = Path(data["split_dir"]) / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Prepared ACDC split is missing: {split_path}; run "
                "python -m fcal_study.prepare_acdc first"
            )
        self.records = []
        with split_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = Record(
                    image_id=int(row["numeric_id"]),
                    image_path=root / row["image_path"],
                    mask_path=root / row["mask_path"],
                    label=int(row["label"]),
                    group_id=int(row["patient_id"]),
                )
                if not record.image_path.is_file() or (
                    self.need_mask and not record.mask_path.is_file()
                ):
                    raise FileNotFoundError(record)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Empty ACDC split: {split_path}")
        self.cache_dir = None
        self._cached_images = None
        self._cached_masks = None

    @staticmethod
    def _square_pad(image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return TF.pad(image, [left, top, side - width - left, side - height - top], fill=0)

    def _geometry(
        self, image: Image.Image, mask: Image.Image | None, generator: torch.Generator,
    ) -> tuple[Image.Image, Image.Image | None]:
        image = self._square_pad(image)
        if mask is not None:
            mask = self._square_pad(mask)
        image = TF.resize(
            image, [self.image_size, self.image_size], InterpolationMode.BILINEAR,
            antialias=True,
        )
        if mask is not None:
            mask = TF.resize(
                mask, [self.image_size, self.image_size], InterpolationMode.NEAREST,
            )
        if self.training:
            angle = float(torch.empty(1).uniform_(-8.0, 8.0, generator=generator))
            maximum_shift = round(self.image_size * 0.04)
            translate = [
                int(torch.randint(-maximum_shift, maximum_shift + 1, (1,), generator=generator)),
                int(torch.randint(-maximum_shift, maximum_shift + 1, (1,), generator=generator)),
            ]
            scale = float(torch.empty(1).uniform_(0.95, 1.05, generator=generator))
            image = TF.affine(
                image, angle, translate, scale, 0.0,
                interpolation=InterpolationMode.BILINEAR, fill=0,
            )
            if mask is not None:
                mask = TF.affine(
                    mask, angle, translate, scale, 0.0,
                    interpolation=InterpolationMode.NEAREST, fill=0,
                )
        return image, mask


class BloodMnistDataset(Dataset):
    """Memory-mapped 224px BloodMNIST+ arrays with classification transforms.

    BloodMNIST supplies class labels but no cell masks, so it deliberately
    supports CE classification only and rejects spatial variants.
    """

    SPLIT_PREFIX: ClassVar[dict[str, str]] = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    def __init__(
        self,
        config: dict,
        split: str,
        need_mask: bool,
        variant: VariantSpec | None = None,
        variants: list[VariantSpec] | None = None,
    ) -> None:
        if need_mask or variant is not None or variants is not None:
            raise ValueError("BloodMNIST has no foreground masks; use CE classification without variants")
        try:
            prefix = self.SPLIT_PREFIX[split]
        except KeyError as error:
            raise ValueError(f"Unsupported BloodMNIST split: {split}") from error
        data = config["data"]
        self.split = split
        self.training = split == "train"
        self.global_seed = int(config["seed"])
        self.image_size = int(data["image_size"])
        self.root = Path(data["medmnist_root"])
        self.image_path = self.root / f"{prefix}_images.npy"
        self.label_path = self.root / f"{prefix}_labels.npy"
        if not self.image_path.is_file() or not self.label_path.is_file():
            raise FileNotFoundError(
                f"Prepared BloodMNIST arrays are missing under {self.root}; "
                "run python -m fcal_study.prepare_bloodmnist"
            )
        labels = np.load(self.label_path, mmap_mode="r")
        self.size = int(labels.shape[0])
        if labels.ndim not in (1, 2) or (labels.ndim == 2 and labels.shape[1] != 1):
            raise ValueError(f"Unexpected BloodMNIST label shape: {labels.shape}")
        self._images = None
        self._labels = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state

    def _open_arrays(self) -> None:
        if self._images is None:
            self._images = np.load(self.image_path, mmap_mode="r")
            self._labels = np.load(self.label_path, mmap_mode="r")
            if self._images.shape[0] != self.size:
                raise ValueError("BloodMNIST image and label counts differ")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | tuple) -> dict[str, torch.Tensor | int]:
        epoch = 0
        if isinstance(index, tuple):
            epoch, index = index
        index = int(index)
        self._open_arrays()
        image = Image.fromarray(np.array(self._images[index], copy=True), mode="RGB")
        generator = torch.Generator().manual_seed(
            (self.global_seed * 1_000_003 + int(epoch) * 11_788 + index) % (2**63 - 1)
        )
        if self.training:
            # Cell identity is rotation invariant. Conservative crops retain the
            # centered cell while adding scale and acquisition-position variety.
            width, height = image.size
            scale = float(torch.empty(1).uniform_(0.85, 1.0, generator=generator))
            crop = max(1, round(min(width, height) * math.sqrt(scale)))
            top = int(torch.randint(0, height - crop + 1, (1,), generator=generator))
            left = int(torch.randint(0, width - crop + 1, (1,), generator=generator))
            image = TF.resized_crop(
                image, top, left, crop, crop, [self.image_size, self.image_size],
                InterpolationMode.BILINEAR, antialias=True,
            )
            if float(torch.rand(1, generator=generator)) < 0.5:
                image = TF.hflip(image)
            if float(torch.rand(1, generator=generator)) < 0.5:
                image = TF.vflip(image)
        elif image.size != (self.image_size, self.image_size):
            image = TF.resize(
                image, [self.image_size, self.image_size],
                InterpolationMode.BILINEAR, antialias=True,
            )
        label = int(np.asarray(self._labels[index]).reshape(-1)[0])
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return {"image": image_tensor, "label": label, "image_id": index}


def dataset_from_config(
    config: dict,
    split: str,
    need_mask: bool,
    variant: VariantSpec | None = None,
    variants: list[VariantSpec] | None = None,
) -> Dataset:
    dataset_name = config["data"].get("dataset", "cub")
    dataset_types = {
        "cub": CubDataset,
        "isic": IsicDataset,
        "oxford_pet": OxfordPetDataset,
        "bloodmnist": BloodMnistDataset,
        "acdc": AcdcCardiacDataset,
    }
    try:
        dataset_type = dataset_types[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from error
    return dataset_type(config, split, need_mask=need_mask, variant=variant, variants=variants)


class EpochTupleSampler(Sampler[tuple[int, int]]):
    """Deterministic epoch-specific shuffle that also seeds per-sample geometry."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        epoch = self.epoch
        generator = torch.Generator().manual_seed(self.seed + epoch)
        self.epoch += 1
        for index in torch.randperm(self.size, generator=generator).tolist():
            yield epoch, index

    def __len__(self) -> int:
        return self.size


class VariantBatchSampler(Sampler[list[tuple[str, int, int]]]):
    def __init__(self, dataset_size: int, variant_count: int, batch_size: int) -> None:
        self.dataset_size = int(dataset_size)
        self.variant_count = int(variant_count)
        self.batch_size = int(batch_size)

    def __iter__(self):
        for variant_index in range(self.variant_count):
            for start in range(0, self.dataset_size, self.batch_size):
                yield [
                    ("variant", variant_index, index)
                    for index in range(start, min(start + self.batch_size, self.dataset_size))
                ]

    def __len__(self) -> int:
        return self.variant_count * math.ceil(self.dataset_size / self.batch_size)


def make_loader(
    config: dict,
    split: str,
    need_mask: bool,
    variant: VariantSpec | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = dataset_from_config(config, split, need_mask=need_mask, variant=variant)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    workers = int(config["data"]["num_workers"])
    options: dict = {}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    sampler = EpochTupleSampler(len(dataset), int(config["seed"])) if split == "train" and shuffle is not False else None
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False if sampler is not None else ((split == "train") if shuffle is None else shuffle),
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
        **options,
    )
    loader.experiment_generator = generator
    return loader


def make_multi_variant_loader(
    config: dict, split: str, variants: list[VariantSpec], batch_size: int = 32,
) -> DataLoader:
    dataset = dataset_from_config(config, split, need_mask=True, variants=variants)
    workers = int(config["data"]["num_workers"])
    options: dict = {}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_sampler=VariantBatchSampler(len(dataset), len(variants), batch_size),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        **options,
    )
    loader.experiment_generator = generator
    return loader
