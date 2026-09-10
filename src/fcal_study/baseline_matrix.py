from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import yaml

from .config import PROJECT_ROOT

ARCHITECTURES = {
    "resnet50": "resnet50_ce.yaml",
    "densenet121": "densenet121_ce.yaml",
    "efficientnet-b0": "efficientnet_b0_ce.yaml",
    "vit-tiny-p16-avg": "vit_tiny_avg_ce.yaml",
    "vit-small-p16-avg": "vit_small_avg_ce.yaml",
    "vit-base-p16-avg": "vit_base_avg_ce.yaml",
}

METHODS = ("rrr", "gain", "camal", "isnet")

SUPPORT = {
    "rrr": set(ARCHITECTURES),
    "gain": set(ARCHITECTURES),
    "camal": set(ARCHITECTURES),
    # The audited in-repository LRP-Flex adapter is ResNet-50-specific.  Do
    # not turn Grad x Input, SiLU/GELU gradients, or attention rollout into an
    # ISNet result by relabeling it.
    "isnet": {"resnet50"},
}

FIXED_RECIPE = {
    "rrr": {
        # Ross et al. reference implementation default (l2_grads).
        "rrr": {"lambda": 1.0e-4},
    },
    "gain": {
        # GAIN paper/released code: sigma=.5, sigmoid scale=8, alpha=1;
        # GAIN-ext external-supervision coefficient omega=10.
        "gain": {
            "sigma": 0.5,
            "omega": 8.0,
            "classification_weight": 1.0,
            "mining_weight": 1.0,
            "supervision_weight": 10.0,
            "epsilon": 1.0e-8,
        },
    },
    "camal": {
        # Hundal et al. Eq. 2 / released implementation.
        "camal": {"coefficient": 1.0, "epsilon": 1.0e-8},
    },
    "isnet": {
        # Selective Faster-ISNet/LRP-Flex released starting values.
        "isnet": {
            "variant": "selective-lrp-flex",
            "lrp_epsilon": 0.01,
            "selective_epsilon": 0.01,
            "heatmap_weight": 0.70,
            "cut_low": 1.0,
            "cut_high": 25.0,
            "background_decay": 0.0,
            "background_scale": 1.0,
            "background_weight": 1.0,
            "foreground_weight": 1.0,
            "z_b_input_rule": False,
            "sanitize_nonfinite_gradients": False,
            "epsilon": 1.0e-10,
        },
    },
}

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "cub" / "baseline_fixed"
DEFAULT_ARCHIVE_OUTPUTS = (
    PROJECT_ROOT.parent
    / "archive"
    / "legacy_projects"
    / "adaptive_fcal_study"
    / "outputs"
)
DEFAULT_ARCHIVE_SPLIT = (
    DEFAULT_ARCHIVE_OUTPUTS.parent / "splits" / "cub200_2011_seed42_val0.2"
)


def supported(method: str, architecture: str) -> bool:
    return architecture in SUPPORT[method]


def config_filename(method: str, architecture: str) -> str:
    label = "gain-ext" if method == "gain" else method
    return f"{architecture}_{label}_fixed_seed42.yaml"


def build_config(method: str, architecture: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown baseline method: {method}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown FCAL architecture: {architecture}")
    if not supported(method, architecture):
        raise ValueError(f"{method} is not validated for {architecture}")
    base_path = PROJECT_ROOT / "configs" / "cub" / ARCHITECTURES[architecture]
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    label = "gain-ext" if method == "gain" else method
    config["project"] = "fcal_research"
    config["run_id"] = f"baseline-fixed/cub/{architecture}-{label}-seed42"
    config["seed"] = 42
    config["method"] = method
    config["training"]["save_epoch5_branch"] = False
    config.update(copy.deepcopy(FIXED_RECIPE[method]))
    if method == "isnet":
        # LRP-Flex manipulates gradients inside forward and rejects AMP.
        config["training"].update({
            "amp": "off",
            "channels_last": False,
            "fused_adamw": False,
        })
    return config


def iter_supported() -> list[tuple[str, str]]:
    return [
        (method, architecture)
        for method in METHODS
        for architecture in ARCHITECTURES
        if supported(method, architecture)
    ]


def write_configs(directory: Path = DEFAULT_CONFIG_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for method, architecture in iter_supported():
        path = directory / config_filename(method, architecture)
        path.write_text(
            yaml.safe_dump(build_config(method, architecture), sort_keys=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def parent_checkpoint(architecture: str, archive_outputs: Path) -> Path:
    return archive_outputs / f"{architecture}-ce-seed42" / "epoch-005-branch.pt"


def training_command(
    method: str,
    architecture: str,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    archive_outputs: Path = DEFAULT_ARCHIVE_OUTPUTS,
    archive_split: Path = DEFAULT_ARCHIVE_SPLIT,
) -> str:
    config = config_dir / config_filename(method, architecture)
    parent = parent_checkpoint(architecture, archive_outputs)
    return (
        f'python -m fcal_study.train "{config}" '
        f'--resume "{parent}" --allow-data-relocation '
        f'--parent-split-dir "{archive_split}"'
    )


def profile_one(
    method: str,
    architecture: str,
    batches: int,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    archive_outputs: Path = DEFAULT_ARCHIVE_OUTPUTS,
) -> dict:
    """Run a bounded, non-checkpointing training smoke test from the CE branch."""
    import torch

    from .config import load_config
    from .data import make_loader
    from .models import create_model
    from .train import train_epoch

    config_path = config_dir / config_filename(method, architecture)
    config = load_config(config_path)
    checkpoint = torch.load(
        parent_checkpoint(architecture, archive_outputs),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = create_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    if config["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=bool(config["training"]["fused_adamw"] and device.type == "cuda"),
    )
    loader = make_loader(config, "train", need_mask=True)

    class LimitedLoader:
        def __len__(self) -> int:
            return min(int(batches), len(loader))

        def __iter__(self):
            return itertools.islice(loader, len(self))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = train_epoch(model, LimitedLoader(), optimizer, config, 5, device)
    metrics["method"] = method
    metrics["architecture"] = architecture
    return metrics


def _print_support() -> None:
    for architecture in ARCHITECTURES:
        values = [method if supported(method, architecture) else "-" for method in METHODS]
        print(f"{architecture:22} " + " ".join(f"{value:7}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and enumerate the fixed-recipe CUB explanation-baseline matrix",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("--directory", type=Path, default=DEFAULT_CONFIG_DIR)
    command_parser = subparsers.add_parser("commands")
    command_parser.add_argument("--directory", type=Path, default=DEFAULT_CONFIG_DIR)
    command_parser.add_argument("--archive-outputs", type=Path, default=DEFAULT_ARCHIVE_OUTPUTS)
    command_parser.add_argument("--archive-split", type=Path, default=DEFAULT_ARCHIVE_SPLIT)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("method", choices=METHODS)
    profile_parser.add_argument("architecture", choices=tuple(ARCHITECTURES))
    profile_parser.add_argument("--batches", type=int, default=2)
    profile_parser.add_argument("--directory", type=Path, default=DEFAULT_CONFIG_DIR)
    profile_parser.add_argument("--archive-outputs", type=Path, default=DEFAULT_ARCHIVE_OUTPUTS)
    subparsers.add_parser("support")
    args = parser.parse_args()

    if args.action == "write":
        for path in write_configs(args.directory):
            print(path)
    elif args.action == "commands":
        if not args.archive_split.is_dir():
            raise FileNotFoundError(f"Missing archived prepared split: {args.archive_split}")
        missing = []
        for method, architecture in iter_supported():
            config = args.directory / config_filename(method, architecture)
            checkpoint = parent_checkpoint(architecture, args.archive_outputs)
            if not config.is_file() or not checkpoint.is_file():
                missing.append((config, checkpoint))
            print(training_command(
                method, architecture, args.directory, args.archive_outputs,
                args.archive_split,
            ))
        if missing:
            raise FileNotFoundError(f"Missing config/checkpoint pairs: {missing}")
    elif args.action == "profile":
        if not supported(args.method, args.architecture):
            raise ValueError(f"{args.method} is not validated for {args.architecture}")
        print(yaml.safe_dump(profile_one(
            args.method,
            args.architecture,
            args.batches,
            args.directory,
            args.archive_outputs,
        ), sort_keys=False))
    else:
        _print_support()


if __name__ == "__main__":
    main()
