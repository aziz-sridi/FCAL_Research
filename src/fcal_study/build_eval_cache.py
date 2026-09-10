from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .data import CubDataset
from .utils import atomic_json, sha256_file


def run(config_path: str) -> Path:
    config = load_config(config_path)
    target = Path(config["data"]["evaluation_cache"])
    if (target / "manifest.json").is_file():
        print(f"Cache already exists: {target}")
        return target
    dataset = CubDataset(config, "test", need_mask=True)
    # Force source loading while the target does not yet exist.
    dataset.cache_dir = None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-building-", dir=target.parent))
    count = len(dataset)
    size = int(config["data"]["image_size"])
    try:
        images = np.lib.format.open_memmap(
            temporary / "images.npy", mode="w+", dtype=np.uint8, shape=(count, size, size, 3)
        )
        masks = np.lib.format.open_memmap(
            temporary / "masks.npy", mode="w+", dtype=np.uint8, shape=(count, size, size)
        )
        image_ids = np.lib.format.open_memmap(
            temporary / "image_ids.npy", mode="w+", dtype=np.int32, shape=(count,)
        )
        labels = np.lib.format.open_memmap(
            temporary / "labels.npy", mode="w+", dtype=np.int16, shape=(count,)
        )
        generator = torch.Generator().manual_seed(int(config["seed"]))
        for index, record in enumerate(dataset.records):
            with Image.open(record.image_path) as source:
                image = source.convert("RGB")
            with Image.open(record.mask_path) as source:
                mask = source.convert("L")
            image, mask = dataset._geometry(image, mask, generator)
            images[index] = np.asarray(image, dtype=np.uint8)
            masks[index] = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8)
            image_ids[index] = record.image_id
            labels[index] = record.label
            if (index + 1) % 1000 == 0:
                print(f"cached={index + 1}/{count}", flush=True)
        del images, masks, image_ids, labels
        files = {}
        for name in ("images.npy", "masks.npy", "image_ids.npy", "labels.npy"):
            path = temporary / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "split": "test",
            "count": count,
            "image_size": size,
            "resize_size": int(config["data"]["resize_size"]),
            "geometry": "bilinear_resize_short_edge_then_center_crop; nearest_mask",
            "files": files,
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
