from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def _official_ids(path: Path) -> set[str]:
    return {
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _materialize(parquet_path: Path, image_dir: Path) -> set[str]:
    ids: set[str] = set()
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=("image_id", "image"), batch_size=64):
        for row in batch.to_pylist():
            image_id = str(row["image_id"])
            if image_id in ids:
                raise RuntimeError(f"Duplicate image ID in {parquet_path}: {image_id}")
            ids.add(image_id)
            payload = row["image"]["bytes"]
            destination = image_dir / f"{image_id}.jpg"
            if not destination.is_file() or destination.stat().st_size != len(payload):
                destination.write_bytes(payload)
    return ids


def materialize(root: Path, train_parquet: Path, test_parquet: Path) -> dict:
    root = root.resolve()
    annotations = root / "annotations"
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    expected_train = _official_ids(annotations / "trainval.txt")
    expected_test = _official_ids(annotations / "test.txt")
    actual_train = _materialize(train_parquet.resolve(), image_dir)
    actual_test = _materialize(test_parquet.resolve(), image_dir)
    if actual_train != expected_train:
        raise RuntimeError(
            f"Mirrored train IDs differ from official trainval IDs: "
            f"missing={sorted(expected_train - actual_train)[:5]}, "
            f"extra={sorted(actual_train - expected_train)[:5]}"
        )
    if actual_test != expected_test:
        raise RuntimeError(
            f"Mirrored test IDs differ from official test IDs: "
            f"missing={sorted(expected_test - actual_test)[:5]}, "
            f"extra={sorted(actual_test - expected_test)[:5]}"
        )
    if actual_train & actual_test:
        raise RuntimeError("Mirrored train and test image IDs overlap")
    for path in image_dir.glob("*.jpg"):
        with Image.open(path) as image:
            image.verify()
    result = {
        "trainval_images": len(actual_train),
        "test_images": len(actual_test),
        "total_images": len(actual_train | actual_test),
        "id_match": True,
        "images_verified": True,
    }
    (root / "mirror_materialization.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the timm Oxford-Pet mirror and verify its IDs against official splits",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.train_parquet, args.test_parquet), indent=2))


if __name__ == "__main__":
    main()
