from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PetRecord:
    name: str
    label: int
    species: int
    breed_id: int
    numeric_id: int


def _read_official(path: Path, ids: dict[str, int]) -> list[PetRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, class_id, species, breed_id = line.split()
        records.append(PetRecord(
            name=name,
            label=int(class_id) - 1,
            species=int(species),
            breed_id=int(breed_id),
            numeric_id=ids[name],
        ))
    return records


def _write_csv(path: Path, records: list[PetRecord], root: Path) -> None:
    columns = (
        "numeric_id", "image_id", "label", "species", "breed_id",
        "image_path", "mask_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.numeric_id):
            image = root / "images" / f"{record.name}.jpg"
            mask = root / "annotations" / "trimaps" / f"{record.name}.png"
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(image if not image.is_file() else mask)
            writer.writerow({
                "numeric_id": record.numeric_id,
                "image_id": record.name,
                "label": record.label,
                "species": record.species,
                "breed_id": record.breed_id,
                "image_path": image.relative_to(root).as_posix(),
                "mask_path": mask.relative_to(root).as_posix(),
            })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path, split_dir: Path, seed: int = 42, validation_fraction: float = 0.20) -> dict:
    root = root.resolve()
    annotations = root / "annotations"
    trainval_file = annotations / "trainval.txt"
    test_file = annotations / "test.txt"
    if not trainval_file.is_file() or not test_file.is_file():
        raise FileNotFoundError("Oxford-IIIT Pet annotations/trainval.txt and test.txt are required")

    names = []
    for path in (trainval_file, test_file):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                names.append(line.split()[0])
    if len(names) != len(set(names)):
        raise RuntimeError("Official Oxford-IIIT Pet trainval/test IDs overlap")
    numeric_ids = {name: index + 1 for index, name in enumerate(sorted(names))}
    trainval = _read_official(trainval_file, numeric_ids)
    test = _read_official(test_file, numeric_ids)
    train, validation = train_test_split(
        trainval,
        test_size=float(validation_fraction),
        random_state=int(seed),
        shuffle=True,
        stratify=[record.label for record in trainval],
    )
    partitions = {"train": list(train), "validation": list(validation), "test": test}
    id_sets = [{record.numeric_id for record in records} for records in partitions.values()]
    if any(id_sets[left] & id_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Prepared Oxford-IIIT Pet splits overlap")

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, records in partitions.items():
        _write_csv(split_dir / f"{name}.csv", records, root)
    manifest = {
        "dataset": "Oxford-IIIT Pet",
        "seed": int(seed),
        "validation_fraction_of_official_trainval": float(validation_fraction),
        "official_test_is_sealed": True,
        "num_classes": 37,
        "counts": {name: len(records) for name, records in partitions.items()},
        "class_counts": {
            name: {str(key): value for key, value in sorted(Counter(r.label for r in records).items())}
            for name, records in partitions.items()
        },
        "mask_rule": "trimap value 2 is background; values 1 and 3 are allowed foreground",
        "source_files": {
            "trainval.txt": _sha256(trainval_file),
            "test.txt": _sha256(test_file),
        },
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-free Oxford-IIIT Pet FCAL splits")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.split_dir, args.seed, args.validation_fraction), indent=2))


if __name__ == "__main__":
    main()
