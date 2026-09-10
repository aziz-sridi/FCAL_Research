from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    for key in (
        "cub_root", "isic_root", "pet_root", "medmnist_root", "cardiac_root",
        "segmentation_root", "split_dir",
        "evaluation_cache",
    ):
        if key not in config["data"]:
            continue
        value = Path(config["data"][key])
        if not value.is_absolute():
            value = (PROJECT_ROOT / value).resolve()
        config["data"][key] = str(value)
    output = Path(config["output_root"])
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    config["output_root"] = str(output)
    return config


def config_hash(config: dict) -> str:
    stable = copy.deepcopy(config)
    stable.pop("_config_path", None)
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_resolved_config(config: dict, path: str | Path) -> None:
    serializable = copy.deepcopy(config)
    serializable.pop("_config_path", None)
    Path(path).write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
