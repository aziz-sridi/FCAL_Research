from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASS_ORDER = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}
FOLDER_TO_CLASS = {
    "melanoma": "MEL",
    "nevus": "NV",
    "basal cell carcinoma": "BCC",
    "actinic keratosis": "AK",
    "pigmented benign keratosis": "BKL",
    # ISIC 2019 represents seborrheic keratosis under the broader BKL class.
    "seborrheic keratosis": "BKL",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
    "squamous cell carcinoma": "SCC",
}
ISIC_PATTERN = re.compile(r"^(ISIC_\d{7})(?:_(?:segmentation|UNET))?$", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    image_path: Path
    class_code: str
    original_split: str


@dataclass(frozen=True)
class PreparedRecord:
    image_id: str
    image_path: Path
    mask_path: Path
    mask_source: str
    class_code: str
    lesion_id: str

    @property
    def numeric_id(self) -> int:
        return int(self.image_id.removeprefix("ISIC_"))

    @property
    def label(self) -> int:
        return CLASS_TO_INDEX[self.class_code]


def _scan_split(dataset_root: Path, split: str) -> list[Candidate]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    records = []
    unknown_folders = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        class_code = FOLDER_TO_CLASS.get(class_dir.name.casefold())
        if class_code is None:
            unknown_folders.append(class_dir.name)
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            if not re.fullmatch(r"ISIC_\d{7}", image_path.stem, re.IGNORECASE):
                raise ValueError(f"Unexpected ISIC filename: {image_path}")
            records.append(Candidate(image_path.stem.upper(), image_path, class_code, split.lower()))
    if unknown_folders:
        raise ValueError(f"Unknown diagnosis folders in {split_root}: {unknown_folders}")
    return records


def _official_labels(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = [column for column in ("image", *CLASS_ORDER, "UNK") if column not in frame.columns]
    if missing:
        raise ValueError(f"Official ground-truth CSV is missing columns: {missing}")
    result = {}
    for row in frame.itertuples(index=False):
        positives = [name for name in (*CLASS_ORDER, "UNK") if float(getattr(row, name)) == 1.0]
        if len(positives) != 1:
            raise ValueError(f"Expected one official label for {row.image}, got {positives}")
        result[str(row.image).upper()] = positives[0]
    return result


def _lesion_ids(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not {"image", "lesion_id"}.issubset(frame.columns):
        raise ValueError("Official metadata CSV must contain image and lesion_id")
    return {str(row.image).upper(): str(row.lesion_id) for row in frame.itertuples(index=False)}


def _mask_index(
    expert_dirs: list[Path], novice_dir: Path, novice_mapping: Path,
) -> tuple[dict[str, tuple[Path, str]], dict[str, int]]:
    index: dict[str, tuple[Path, str]] = {}
    for directory in expert_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*.png")):
            match = ISIC_PATTERN.fullmatch(path.stem)
            if match:
                index[match.group(1).upper()] = (path, "expert_challenge")
    expert_count = len(index)
    if not novice_dir.is_dir():
        raise FileNotFoundError(novice_dir)
    mapping = pd.read_csv(novice_mapping, dtype=str)
    if not {"Filename", "ISIC_ID"}.issubset(mapping.columns):
        raise ValueError("Novice mask mapping must contain Filename and ISIC_ID")
    for row in mapping.itertuples(index=False):
        image_id = str(row.ISIC_ID).upper()
        path = novice_dir / str(row.Filename)
        if image_id not in index and path.is_file():
            index[image_id] = (path, "reviewed_novice_unet")
    return index, {
        "expert_ids": expert_count,
        "reviewed_novice_ids_added": len(index) - expert_count,
        "total_mask_ids": len(index),
    }


def _resolve_candidates(
    candidates: list[Candidate], official: dict[str, str], audit: Counter,
) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.image_id].append(candidate)
    resolved = []
    for image_id, options in sorted(grouped.items()):
        if len(options) > 1:
            audit["duplicate_file_copies_collapsed"] += len(options) - 1
        official_class = official.get(image_id)
        if official_class == "UNK":
            audit["official_unknown_excluded"] += 1
            continue
        matching = [item for item in options if not official_class or item.class_code == official_class]
        if matching:
            resolved.append(min(matching, key=lambda item: str(item.image_path)))
            continue
        classes = {item.class_code for item in options}
        if official_class:
            audit["folder_official_label_conflicts_excluded"] += 1
        elif len(classes) > 1:
            audit["ambiguous_folder_labels_excluded"] += 1
        else:
            resolved.append(min(options, key=lambda item: str(item.image_path)))
    return resolved


def _attach_masks(
    candidates: list[Candidate], masks: dict[str, tuple[Path, str]],
    lesion_ids: dict[str, str], audit: Counter,
) -> list[PreparedRecord]:
    records = []
    for candidate in candidates:
        match = masks.get(candidate.image_id)
        if match is None:
            audit["missing_lesion_mask_excluded"] += 1
            continue
        records.append(PreparedRecord(
            image_id=candidate.image_id,
            image_path=candidate.image_path,
            mask_path=match[0],
            mask_source=match[1],
            class_code=candidate.class_code,
            lesion_id=lesion_ids.get(candidate.image_id, ""),
        ))
    return records


def _copy_mask(record: PreparedRecord, output_root: Path) -> str:
    destination = output_root / f"{record.image_id}.png"
    if destination.is_file():
        return destination.name
    with Image.open(record.image_path) as image_source:
        image_size = image_source.size
    with Image.open(record.mask_path) as mask_source:
        mask = mask_source.convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        values = np.asarray(mask)
    binary = (values >= 128).astype(np.uint8) * 255
    if not bool(binary.any()):
        raise ValueError(f"Empty lesion mask: {record.mask_path}")
    Image.fromarray(binary, mode="L").save(destination, optimize=True)
    return destination.name


def _write_split(
    path: Path, records: list[PreparedRecord], dataset_root: Path, mask_root: Path,
) -> None:
    fields = (
        "image_id", "numeric_id", "image_path", "mask_path", "label", "class_code",
        "lesion_id", "mask_source",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.image_id):
            mask_name = _copy_mask(record, mask_root)
            writer.writerow({
                "image_id": record.image_id,
                "numeric_id": record.numeric_id,
                "image_path": record.image_path.relative_to(dataset_root).as_posix(),
                "mask_path": mask_name,
                "label": record.label,
                "class_code": record.class_code,
                "lesion_id": record.lesion_id,
                "mask_source": record.mask_source,
            })
    path.with_suffix(".txt").write_text(
        "".join(f"{record.numeric_id}\n" for record in sorted(records, key=lambda item: item.image_id)),
        encoding="utf-8",
    )


def _class_counts(records: list[PreparedRecord]) -> dict[str, int]:
    counts = Counter(record.class_code for record in records)
    return {name: counts[name] for name in CLASS_ORDER}


def prepare(
    dataset_root: Path,
    metadata_csv: Path,
    ground_truth_csv: Path,
    expert_mask_dirs: list[Path],
    novice_mask_dir: Path,
    novice_mapping: Path,
    split_dir: Path,
    mask_output_dir: Path,
    seed: int = 42,
) -> dict:
    dataset_root = dataset_root.resolve()
    split_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    official = _official_labels(ground_truth_csv)
    lesions = _lesion_ids(metadata_csv)
    mask_index, mask_audit = _mask_index(expert_mask_dirs, novice_mask_dir, novice_mapping)
    audit: Counter = Counter()

    test_raw = _scan_split(dataset_root, "Test")
    train_raw = _scan_split(dataset_root, "Train")
    test = _resolve_candidates(test_raw, official, audit)
    train_without_test_ids = [item for item in train_raw if item.image_id not in {x.image_id for x in test}]
    audit["train_exact_test_id_copies_excluded"] = len(train_raw) - len(train_without_test_ids)
    train = _resolve_candidates(train_without_test_ids, official, audit)

    test_lesions = {lesions[item.image_id] for item in test if lesions.get(item.image_id)}
    leakage_free_train = []
    for item in train:
        lesion_id = lesions.get(item.image_id, "")
        if lesion_id and lesion_id in test_lesions:
            audit["train_test_lesion_siblings_excluded"] += 1
        else:
            leakage_free_train.append(item)

    masked_train = _attach_masks(leakage_free_train, mask_index, lesions, audit)
    masked_test = _attach_masks(test, mask_index, lesions, audit)
    labels = np.asarray([record.label for record in masked_train])
    groups = np.asarray([record.lesion_id or record.image_id for record in masked_train])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(splitter.split(np.zeros(len(labels)), labels, groups))
    final_train = [masked_train[index] for index in train_indices]
    validation = [masked_train[index] for index in validation_indices]

    split_records = {"train": final_train, "validation": validation, "test": masked_test}
    id_sets = {name: {record.image_id for record in records} for name, records in split_records.items()}
    if id_sets["train"] & id_sets["validation"] or id_sets["train"] & id_sets["test"] or id_sets["validation"] & id_sets["test"]:
        raise RuntimeError("Prepared ISIC image IDs are not disjoint")
    lesion_sets = {
        name: {record.lesion_id for record in records if record.lesion_id}
        for name, records in split_records.items()
    }
    if lesion_sets["train"] & lesion_sets["validation"] or lesion_sets["train"] & lesion_sets["test"] or lesion_sets["validation"] & lesion_sets["test"]:
        raise RuntimeError("Prepared ISIC lesion IDs are not disjoint")

    for name, records in split_records.items():
        _write_split(split_dir / f"{name}.csv", records, dataset_root, mask_output_dir)
    manifest = {
        "dataset": "ISIC skin-cancer classification subset",
        "seed": seed,
        "class_order": list(CLASS_ORDER),
        "class_mapping": CLASS_TO_INDEX,
        "seborrheic_keratosis_policy": "merged_into_BKL",
        "mask_policy": "official challenge mask preferred, then reviewed novice U-Net mask",
        "mask_sources": {
            "expert": "ISIC 2016/2017/2018 Task 1 challenge ground truth",
            "reviewed_novice": "ISIC Archive DOI 10.34970/387951",
        },
        "raw_counts": {"train_files": len(train_raw), "test_files": len(test_raw)},
        "exclusions": dict(sorted(audit.items())),
        "mask_index": mask_audit,
        "splits": {
            name: {"samples": len(records), "class_counts": _class_counts(records)}
            for name, records in split_records.items()
        },
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-free, mask-covered ISIC FCAL splits")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--expert-mask-dir", type=Path, action="append", default=[])
    parser.add_argument("--novice-mask-dir", type=Path, required=True)
    parser.add_argument("--novice-mapping", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--mask-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(
        args.dataset_root, args.metadata_csv, args.ground_truth_csv,
        args.expert_mask_dir, args.novice_mask_dir, args.novice_mapping,
        args.split_dir, args.mask_output_dir, args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
