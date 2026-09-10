"""Build figures and artifact-driven tables used by the FCAL V1/V2 article."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "reports" / "v1_v2_article"
VIS_CACHE = ROOT / "reports" / "simple_visualization_cache"
EVAL_CACHE = ROOT / "cache" / "cub_test_224_center"
FIXED_RESULTS = ROOT / "results" / "baseline_fixed" / "cub"
TUNED_RESULTS = ROOT / "results" / "tuned" / "cub"
GENERATED_TABLES = ROOT / "paper" / "generated"

sys.path.insert(0, str(ROOT / "src"))
from fcal_study.complete_results import ARCHITECTURES as RESULT_ARCHITECTURES
from fcal_study.complete_results import GENERIC_BACKGROUND
from fcal_study.variants import apply_variant, parse_variant

ARCHITECTURES = {
    "ResNet-50": ("resnet50-ce-seed42", "resnet50-fcal-seed42", "resnet50-fcal-v2-seed42"),
    "DenseNet-121": ("densenet121-ce-seed42", "densenet121-fcal-seed42", "densenet121-fcal-v2-seed42"),
    "EfficientNet-B0": ("efficientnet-b0-ce-seed42", "efficientnet-b0-fcal-seed42", "efficientnet-b0-fcal-v2-seed42"),
    "ViT-B/16 avg": ("vit-base-p16-avg-ce-seed42", "vit-base-p16-avg-fcal-seed42", "vit-base-p16-avg-fcal-v2-seed42"),
    "ViT-S/16 avg": ("vit-small-p16-avg-ce-seed42", "vit-small-p16-avg-fcal-seed42", "vit-small-p16-avg-fcal-v2-seed42"),
    "ViT-T/16 avg": ("vit-tiny-p16-avg-ce-seed42", "vit-tiny-p16-avg-fcal-seed42", "vit-tiny-p16-avg-fcal-v2-seed42"),
}
METHODS = ("CE", "V1", "V2")
COLORS = ("#66788A", "#D98932", "#2F8F64")

GENERAL_BASELINES = ("CAMAL", "GAIN-ext", "RRR")
GENERAL_TUNED_METHODS = ("FCAL V2", *GENERAL_BASELINES)
ALL_FIXED_METHODS = ("CE", *GENERAL_TUNED_METHODS, "Faster-ISNet")
ARCHITECTURE_SLUGS = tuple(slug for slug, _, _ in RESULT_ARCHITECTURES)
ARCHITECTURE_NAMES = tuple(name for _, name, _ in RESULT_ARCHITECTURES)
ARTICLE_METRICS = (
    ("clean_accuracy", "Clean", True),
    ("bg_mean_accuracy", "BG mean", True),
    ("bg_worst_accuracy", "BG worst", True),
    ("foreground_only_accuracy", "FG only", True),
    ("strict_leakage", "Leak $\\downarrow$", False),
)
WTL_METRICS = tuple(metric for metric, _, _ in ARTICLE_METRICS)

FIXED_ARTIFACTS = {
    "metrics": "comparison_metrics.csv",
    "wtl": "architecture_win_tie_loss.csv",
    "efficiency": "efficiency_runs.csv",
    "overhead": "efficiency_overhead.csv",
}
TUNED_ARTIFACTS = {
    "selected": "validation_selected_hyperparameters.csv",
    "metrics": "tuned_winner_metrics.csv",
    "attempts": "tuning_attempts.csv",
    "accounting": "tuning_gpu_hour_accounting.csv",
    "fixed_tuned_wtl": "fixed_vs_tuned_architecture_win_tie_loss.csv",
    "tuned_wtl": "tuned_architecture_win_tie_loss.csv",
}

QUALITATIVE_TESTS = (
    ("Clean", "clean"),
    ("Blur (s3)", "bg_blur_s3"),
    ("Color jitter (s3)", "bg_color_jitter_s3"),
    ("Grayscale", "bg_grayscale_s1"),
    ("Gaussian noise (s3)", "bg_gaussian_noise_s3"),
    ("Solid mean", "bg_solid_mean_s1"),
    ("Random solid", "bg_random_solid_s1"),
    ("Random RGB", "bg_random_rgb_s1"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_required_csvs(root: Path, names: dict[str, str]) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {key: root / filename for key, filename in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Final article comparison artifacts are incomplete; missing: " + ", ".join(missing)
        )
    return {key: pd.read_csv(path) for key, path in paths.items()}, paths


def _require_columns(frame: pd.DataFrame, columns: set[str], identity: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{identity} is missing columns: {', '.join(missing)}")


def _expected_cells(methods: tuple[str, ...], include_isnet: bool = False) -> set[tuple[str, str]]:
    cells = {(architecture, method) for architecture in ARCHITECTURE_SLUGS for method in methods}
    if include_isnet:
        cells.add(("resnet50", "Faster-ISNet"))
    return cells


def _validate_cells(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    identity: str,
    include_isnet: bool = False,
) -> None:
    _require_columns(frame, {"architecture_slug", "method"}, identity)
    selected = frame.loc[frame["method"].isin((*methods, "Faster-ISNet"))]
    if selected.duplicated(["architecture_slug", "method"]).any():
        raise ValueError(f"{identity} has duplicate architecture/method rows")
    observed = set(zip(selected["architecture_slug"], selected["method"]))
    expected = _expected_cells(methods, include_isnet)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            f"{identity} does not match the declared support matrix; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_wtl(
    frame: pd.DataFrame,
    *,
    references: tuple[str, ...],
    candidates: tuple[str, ...],
    identity: str,
) -> None:
    required = {"metric", "reference", "candidate", "architectures", "wins", "ties", "losses"}
    _require_columns(frame, required, identity)
    selected = frame.loc[
        frame["reference"].isin(references)
        & frame["candidate"].isin(candidates)
        & frame["metric"].isin(WTL_METRICS)
    ]
    expected = {(reference, candidate, metric) for reference in references for candidate in candidates for metric in WTL_METRICS}
    observed = set(zip(selected["reference"], selected["candidate"], selected["metric"]))
    if selected.duplicated(["reference", "candidate", "metric"]).any() or observed != expected:
        raise ValueError(f"{identity} is missing or duplicates required W/T/L rows")
    if not selected["architectures"].eq(len(ARCHITECTURE_SLUGS)).all():
        raise ValueError(f"{identity} must report exactly six eligible architectures")


def _tex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _table(
    *,
    caption: str,
    label: str,
    columns: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    wide: bool = False,
) -> str:
    environment = "table*" if wide else "table"
    body = [
        f"\\begin{{{environment}}}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend((r"\bottomrule", r"\end{tabular}", f"\\end{{{environment}}}", ""))
    return "\n".join(body)


def _metric_value(metric: str, value: float) -> str:
    return f"{100.0 * value:.2f}" if metric != "strict_leakage" else f"{value:.3f}"


def _metric_summary_table(metrics: pd.DataFrame, *, tuned: bool) -> str:
    methods = GENERAL_TUNED_METHODS if tuned else ("CE", *GENERAL_TUNED_METHODS)
    rows = []
    for method in methods:
        group = metrics.loc[metrics["method"].eq(method)]
        rows.append((
            _tex_escape(method),
            "six-architecture mean",
            *(_metric_value(metric, float(group[metric].mean())) for metric, _, _ in ARTICLE_METRICS),
        ))
    isnet = metrics.loc[metrics["method"].eq("Faster-ISNet")].iloc[0]
    rows.append((
        "Faster-ISNet$^{\\dagger}$",
        "ResNet-50 only",
        *(_metric_value(metric, float(isnet[metric])) for metric, _, _ in ARTICLE_METRICS),
    ))
    arm = "Tuned" if tuned else "Fixed/default"
    return _table(
        caption=(
            f"{arm} CUB comparison. Accuracy entries are percentages. General methods are "
            "six-architecture means; $^{\\dagger}$Faster-ISNet is support-restricted and its "
            "single ResNet-50 value must not be compared as an architecture mean."
        ),
        label=f"tab:{'tuned' if tuned else 'fixed'}-comparison-generated",
        columns="llrrrrr",
        headers=("Method", "Scope", *(label for _, label, _ in ARTICLE_METRICS)),
        rows=rows,
        wide=True,
    )


def _fcal_wtl_table(frame: pd.DataFrame, *, tuned: bool) -> str:
    rows = []
    for candidate in GENERAL_BASELINES:
        subset = frame.loc[
            frame["reference"].eq("FCAL V2") & frame["candidate"].eq(candidate)
        ].set_index("metric")
        values = []
        for metric in WTL_METRICS:
            row = subset.loc[metric]
            # Source rows count candidate performance.  Reverse wins/losses so
            # every displayed cell consistently means FCAL V2 W/T/L.
            values.append(f"{int(row.losses)}/{int(row.ties)}/{int(row.wins)}")
        rows.append((_tex_escape(candidate), *values))
    arm = "validation-selected tuned" if tuned else "fixed/default"
    return _table(
        caption=(
            f"FCAL V2 architecture wins/ties/losses against each {arm} competitor. "
            "All counts use the same six architectures. Faster-ISNet is excluded because only "
            "ResNet-50 is supported."
        ),
        label=f"tab:{'tuned' if tuned else 'fixed'}-wtl-generated",
        columns="l" + "r" * len(ARTICLE_METRICS),
        headers=("Competitor", *(label.replace(" $\\downarrow$", "") for _, label, _ in ARTICLE_METRICS)),
        rows=rows,
        wide=True,
    )


def _fixed_efficiency_table(efficiency: pd.DataFrame, overhead: pd.DataFrame) -> str:
    rows = []
    for method in GENERAL_TUNED_METHODS:
        runs = efficiency.loc[efficiency["method"].eq(method)]
        if method == "FCAL V2":
            post_overhead = vram_overhead = "reference"
        else:
            paired = overhead.loc[
                overhead["reference"].eq("FCAL V2") & overhead["candidate"].eq(method)
            ]
            post_overhead = f"{paired['post_branch_wall_seconds_overhead_percent'].median():+.1f}\\%"
            vram_overhead = f"{paired['peak_cuda_memory_gib_overhead_percent'].median():+.1f}\\%"
        rows.append((
            _tex_escape(method), "six architectures",
            f"{runs['post_branch_wall_seconds'].median() / 60.0:.1f}", post_overhead, vram_overhead,
        ))
    runs = efficiency.loc[efficiency["method"].eq("Faster-ISNet")]
    paired = overhead.loc[
        overhead["reference"].eq("FCAL V2") & overhead["candidate"].eq("Faster-ISNet")
    ]
    rows.append((
        "Faster-ISNet$^{\\dagger}$", "ResNet-50 only",
        f"{runs['post_branch_wall_seconds'].iloc[0] / 60.0:.1f}",
        f"{paired['post_branch_wall_seconds_overhead_percent'].iloc[0]:+.1f}\\%",
        f"{paired['peak_cuda_memory_gib_overhead_percent'].iloc[0]:+.1f}\\%",
    ))
    return _table(
        caption=(
            "Fixed/default training efficiency. Times and overheads are medians across six "
            "architectures except $^{\\dagger}$Faster-ISNet, which is ResNet-50 only. "
            "Overheads use the matched FCAL V2 run within each architecture."
        ),
        label="tab:fixed-efficiency-generated",
        columns="llrrr",
        headers=("Method", "Scope", "Post-branch min", "Time vs V2", "VRAM vs V2"),
        rows=rows,
        wide=True,
    )


def _selected_hyperparameters_table(selected: pd.DataFrame) -> str:
    indexed = selected.set_index(["architecture_slug", "method"])
    rows = []
    names = {slug: name for slug, name, _ in RESULT_ARCHITECTURES}
    display_methods = ("FCAL V2", "RRR", "GAIN-ext", "CAMAL")
    for architecture in ARCHITECTURE_SLUGS:
        values = []
        for method in display_methods:
            values.append(f"{float(indexed.loc[(architecture, method), 'value']):g}")
        isnet = (
            f"{float(indexed.loc[(architecture, 'Faster-ISNet'), 'value']):g}"
            if (architecture, "Faster-ISNet") in indexed.index else "--"
        )
        rows.append((_tex_escape(names[architecture]), *values, isnet))
    coefficients = (
        selected[["method", "coefficient"]]
        .drop_duplicates()
        .sort_values("method")
    )
    coefficient_note = "; ".join(
        f"{_tex_escape(row.method)}: \\texttt{{{_tex_escape(row.coefficient)}}}"
        for row in coefficients.itertuples(index=False)
    )
    return _table(
        caption=(
            "Validation-selected coefficient values from the equal-budget successive-halving arm. "
            "A dash is structural non-support, not a missing result. Coefficient keys: "
            f"{coefficient_note}."
        ),
        label="tab:tuned-hyperparameters-generated",
        columns="lrrrrr",
        headers=("Architecture", "FCAL V2", "RRR", "GAIN-ext", "CAMAL", "Faster-ISNet$^{\\dagger}$"),
        rows=rows,
        wide=True,
    )


def _tuning_accounting_table(accounting: pd.DataFrame) -> str:
    method_rows = accounting.loc[accounting["scope"].eq("method")].set_index("method")
    rows = []
    for method in (*GENERAL_TUNED_METHODS, "Faster-ISNet"):
        row = method_rows.loc[method]
        scope = "ResNet-50 only" if method == "Faster-ISNet" else "six architectures"
        label = "Faster-ISNet$^{\\dagger}$" if method == "Faster-ISNet" else _tex_escape(method)
        rows.append((
            label, scope, str(int(row.attempts)), str(int(row.failed_attempts)),
            f"{row.gpu_hours_including_failures:.3f}", f"{row.failed_gpu_hours:.3f}",
        ))
    study = accounting.loc[accounting["scope"].eq("study")]
    if len(study) != 1:
        raise ValueError("Tuning accounting must contain exactly one study-total row")
    total = study.iloc[0]
    rows.append((
        "Total", "declared support matrix", str(int(total.attempts)),
        str(int(total.failed_attempts)), f"{total.gpu_hours_including_failures:.3f}",
        f"{total.failed_gpu_hours:.3f}",
    ))
    return _table(
        caption=(
            "Realized validation-only tuning cost under allocated-wall-clock GPU accounting. "
            "All attempts, including failures, are charged. $^{\\dagger}$Faster-ISNet covers "
            "ResNet-50 only."
        ),
        label="tab:tuning-cost-generated",
        columns="llrrrr",
        headers=("Method", "Scope", "Attempts", "Failures", "GPU h", "Failed GPU h"),
        rows=rows,
        wide=True,
    )


def _fixed_tuned_wtl_table(frame: pd.DataFrame) -> str:
    rows = []
    for method in GENERAL_TUNED_METHODS:
        subset = frame.loc[frame["method"].eq(method)].set_index("metric")
        rows.append((
            _tex_escape(method),
            *(f"{int(subset.loc[metric, 'wins'])}/{int(subset.loc[metric, 'ties'])}/"
              f"{int(subset.loc[metric, 'losses'])}" for metric in WTL_METRICS),
        ))
    return _table(
        caption=(
            "Tuned-versus-fixed architecture wins/ties/losses for each method across the six "
            "eligible architectures. Faster-ISNet is excluded because its support is ResNet-50 only."
        ),
        label="tab:fixed-tuned-wtl-generated",
        columns="l" + "r" * len(ARTICLE_METRICS),
        headers=("Method", *(label.replace(" $\\downarrow$", "") for _, label, _ in ARTICLE_METRICS)),
        rows=rows,
        wide=True,
    )


def build_comparison_tables(
    fixed_root: Path = FIXED_RESULTS,
    tuned_root: Path = TUNED_RESULTS,
    output_root: Path = GENERATED_TABLES,
) -> tuple[Path, ...]:
    """Generate manuscript-ready TeX only from complete final CSV artifacts."""
    fixed, fixed_paths = _read_required_csvs(Path(fixed_root), FIXED_ARTIFACTS)
    tuned, tuned_paths = _read_required_csvs(Path(tuned_root), TUNED_ARTIFACTS)

    metric_columns = {metric for metric, _, _ in ARTICLE_METRICS}
    _require_columns(fixed["metrics"], metric_columns, "fixed comparison metrics")
    _require_columns(tuned["metrics"], metric_columns, "tuned winner metrics")
    _validate_cells(
        fixed["metrics"], methods=("CE", *GENERAL_TUNED_METHODS),
        identity="fixed comparison metrics", include_isnet=True,
    )
    _validate_cells(
        tuned["metrics"], methods=GENERAL_TUNED_METHODS,
        identity="tuned winner metrics", include_isnet=True,
    )
    _validate_cells(
        tuned["selected"], methods=GENERAL_TUNED_METHODS,
        identity="validation-selected hyperparameters", include_isnet=True,
    )
    _require_columns(tuned["selected"], {"coefficient", "value", "selection_split"}, "selected hyperparameters")
    if not tuned["selected"]["selection_split"].eq("validation").all():
        raise ValueError("Every selected hyperparameter must declare validation selection")

    _validate_wtl(
        fixed["wtl"], references=("FCAL V2",), candidates=GENERAL_BASELINES,
        identity="fixed architecture W/T/L",
    )
    _validate_wtl(
        tuned["tuned_wtl"], references=("FCAL V2",), candidates=GENERAL_BASELINES,
        identity="tuned architecture W/T/L",
    )
    _require_columns(
        tuned["fixed_tuned_wtl"],
        {"method", "metric", "architectures", "wins", "ties", "losses"},
        "fixed-versus-tuned W/T/L",
    )
    fixed_tuned = tuned["fixed_tuned_wtl"].loc[
        tuned["fixed_tuned_wtl"]["method"].isin(GENERAL_TUNED_METHODS)
        & tuned["fixed_tuned_wtl"]["metric"].isin(WTL_METRICS)
    ]
    expected_fixed_tuned = {(method, metric) for method in GENERAL_TUNED_METHODS for metric in WTL_METRICS}
    observed_fixed_tuned = set(zip(fixed_tuned["method"], fixed_tuned["metric"]))
    if observed_fixed_tuned != expected_fixed_tuned or not fixed_tuned["architectures"].eq(6).all():
        raise ValueError("Fixed-versus-tuned W/T/L is incomplete or not six-architecture")

    _require_columns(
        fixed["efficiency"], {"architecture_slug", "method", "post_branch_wall_seconds"},
        "fixed efficiency runs",
    )
    _validate_cells(
        fixed["efficiency"], methods=("CE", *GENERAL_TUNED_METHODS),
        identity="fixed efficiency runs", include_isnet=True,
    )
    _require_columns(
        fixed["overhead"],
        {"reference", "candidate", "post_branch_wall_seconds_overhead_percent", "peak_cuda_memory_gib_overhead_percent"},
        "fixed efficiency overhead",
    )
    efficiency_expectations = tuple((method, 6) for method in GENERAL_BASELINES) + (
        ("Faster-ISNet", 1),
    )
    for method, expected_count in efficiency_expectations:
        rows = fixed["overhead"].loc[
            fixed["overhead"]["reference"].eq("FCAL V2")
            & fixed["overhead"]["candidate"].eq(method)
        ]
        if len(rows) != expected_count:
            raise ValueError(f"Fixed FCAL V2 efficiency pairing is incomplete for {method}")

    _require_columns(
        tuned["accounting"],
        {"scope", "method", "attempts", "failed_attempts", "gpu_hours_including_failures", "failed_gpu_hours"},
        "tuning GPU-hour accounting",
    )
    method_accounting = tuned["accounting"].loc[tuned["accounting"]["scope"].eq("method")]
    if set(method_accounting["method"]) != set(GENERAL_TUNED_METHODS).union({"Faster-ISNet"}):
        raise ValueError("Tuning GPU-hour accounting does not cover all five methods")
    _require_columns(tuned["attempts"], {"status", "gpu_hours", "method"}, "tuning attempts")
    attempts_total = float(tuned["attempts"]["gpu_hours"].sum())
    study_total = tuned["accounting"].loc[tuned["accounting"]["scope"].eq("study")]
    if len(study_total) != 1 or not np.isclose(
        attempts_total, float(study_total.iloc[0]["gpu_hours_including_failures"]), atol=1e-12
    ):
        raise ValueError("Tuning attempt GPU hours do not reconcile with the study total")

    rendered = {
        "fixed_comparison_table.tex": _metric_summary_table(fixed["metrics"], tuned=False),
        "fixed_wtl_table.tex": _fcal_wtl_table(fixed["wtl"], tuned=False),
        "fixed_efficiency_table.tex": _fixed_efficiency_table(fixed["efficiency"], fixed["overhead"]),
        "tuned_comparison_table.tex": _metric_summary_table(tuned["metrics"], tuned=True),
        "tuned_wtl_table.tex": _fcal_wtl_table(tuned["tuned_wtl"], tuned=True),
        "tuned_hyperparameters_table.tex": _selected_hyperparameters_table(tuned["selected"]),
        "tuning_cost_table.tex": _tuning_accounting_table(tuned["accounting"]),
        "fixed_vs_tuned_wtl_table.tex": _fixed_tuned_wtl_table(fixed_tuned),
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, content in rendered.items():
        path = output_root / filename
        path.write_text(content, encoding="utf-8")
        outputs.append(path)
    source_paths = {f"fixed/{key}": path for key, path in fixed_paths.items()}
    source_paths.update({f"tuned/{key}": path for key, path in tuned_paths.items()})
    manifest_path = output_root / "comparison_tables_manifest.json"
    manifest_path.write_text(json.dumps({
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in sorted(source_paths.items())
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in outputs
        },
        "support_restriction": {
            "method": "Faster-ISNet",
            "architectures": ["resnet50"],
            "architecture_wtl_eligible": False,
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    outputs.append(manifest_path)
    return tuple(outputs)


def load_metrics(run_id: str) -> dict[str, float]:
    path = OUTPUTS / run_id / "evaluation" / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_name = {row["variant"]: row for row in rows}
    background = [by_name[key]["accuracy"] for key in GENERIC_BACKGROUND]
    clean = by_name["clean"]
    return {
        "Clean": 100.0 * clean["accuracy"],
        "BG mean": 100.0 * float(np.mean(background)),
        "BG worst": 100.0 * float(np.min(background)),
        "FG-only": 100.0 * by_name["foreground_only"]["accuracy"],
        "Leakage": float(clean["strict_leakage"]),
    }


def class_names() -> list[str]:
    path = ROOT.parent / "cub2002011" / "versions" / "7" / "CUB_200_2011" / "classes.txt"
    return [
        line.split(maxsplit=1)[1].split(".", 1)[-1].replace("_", " ")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def cached_image(image_id: int, variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    image_ids = np.load(EVAL_CACHE / "image_ids.npy", mmap_mode="r")
    positions = np.flatnonzero(image_ids == image_id)
    if len(positions) != 1:
        raise ValueError(f"Expected one cache entry for image {image_id}")
    position = int(positions[0])
    image = torch.from_numpy(
        np.array(np.load(EVAL_CACHE / "images.npy", mmap_mode="r")[position], copy=True)
    ).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(
        np.array(np.load(EVAL_CACHE / "masks.npy", mmap_mode="r")[position], copy=True)
    ).unsqueeze(0).float()
    label = int(np.load(EVAL_CACHE / "labels.npy", mmap_mode="r")[position])
    if variant == "clean":
        tested = image.clone()
    else:
        name, marker, severity = variant.rpartition("_s")
        if marker != "_s" or not severity.isdigit():
            raise ValueError(f"Expected cached variant key ending in _sN, got {variant}")
        tested = apply_variant(image, mask, image_id, parse_variant(f"{name}:{severity}"), 42)
    return image, tested, mask, label


def cached_method(
    slug: str,
    method: str,
    variant: str,
    image_id: int,
) -> dict[str, np.ndarray | float | int]:
    path = VIS_CACHE / slug / variant / f"{method}.npz"
    with np.load(path) as archive:
        position = int(np.flatnonzero(archive["image_id"] == image_id)[0])
        return {
            "prediction": int(archive["prediction"][position]),
            "confidence": float(archive["confidence"][position]),
            "cam": archive["cam"][position].astype(np.float32),
        }


def normalized_cam(cam: np.ndarray) -> torch.Tensor:
    """Apply one robust normalization rule to every CNN and ViT map."""
    attention = torch.from_numpy(cam).float()[None, None]
    attention = F.interpolate(attention, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    finite = attention[torch.isfinite(attention)]
    if finite.numel() == 0:
        return torch.zeros_like(attention)
    low = torch.quantile(finite, 0.02)
    high = torch.quantile(finite, 0.98)
    if float(high - low) <= 1e-8:
        return torch.zeros_like(attention)
    return ((attention - low) / (high - low)).clamp(0, 1)


def overlay_cam(axis, image: torch.Tensor, mask: torch.Tensor, cam: np.ndarray) -> None:
    attention = normalized_cam(cam)
    axis.imshow(image.permute(1, 2, 0))
    axis.imshow(attention, cmap="jet", alpha=0.56, vmin=0, vmax=1)
    axis.contour(mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)


def select_rescue(slug: str, variant: str) -> int:
    """Pick the clearest verified CE-wrong to V2-correct case in the saved cache."""
    baseline_path = VIS_CACHE / slug / variant / "baseline.npz"
    v2_path = VIS_CACHE / slug / variant / "v2.npz"
    with np.load(baseline_path) as baseline, np.load(v2_path) as v2:
        v2_positions = {int(image_id): i for i, image_id in enumerate(v2["image_id"])}
        candidates: list[tuple[float, int]] = []
        for baseline_position, image_id_value in enumerate(baseline["image_id"]):
            image_id = int(image_id_value)
            v2_position = v2_positions.get(image_id)
            if v2_position is None:
                continue
            _, _, mask, label = cached_image(image_id, variant)
            baseline_prediction = int(baseline["prediction"][baseline_position])
            v2_prediction = int(v2["prediction"][v2_position])
            if baseline_prediction == label or v2_prediction != label:
                continue

            background = 1.0 - mask[0]
            baseline_attention = normalized_cam(baseline["cam"][baseline_position])
            v2_attention = normalized_cam(v2["cam"][v2_position])
            baseline_leakage = float((baseline_attention * background).sum() / baseline_attention.sum().clamp_min(1e-8))
            v2_leakage = float((v2_attention * background).sum() / v2_attention.sum().clamp_min(1e-8))
            confidence = float(v2["confidence"][v2_position])
            candidates.append((baseline_leakage - v2_leakage + 0.10 * confidence, image_id))

    if not candidates:
        raise ValueError(f"No verified CE-wrong/V2-correct rescue cached for {slug}/{variant}")
    return max(candidates)[1]


def rescue_family_figure(
    model_name: str,
    slug: str,
    title: str,
    output_name: str,
) -> None:
    names = class_names()
    wrapped = lambda value: textwrap.fill(value, width=20)
    fig, axes = plt.subplots(4, 6, figsize=(13.2, 9.0), constrained_layout=True)
    for test_index, (test_name, variant) in enumerate(QUALITATIVE_TESTS):
        row = test_index // 2
        column = 3 * (test_index % 2)
        image_id = select_rescue(slug, variant)
        _, tested, mask, label = cached_image(image_id, variant)
        baseline = cached_method(slug, "baseline", variant, image_id)
        v2 = cached_method(slug, "v2", variant, image_id)
        print(f"{model_name}: {test_name}: image {image_id}")

        axes[row, column].imshow(tested.permute(1, 2, 0))
        axes[row, column].contour(mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)
        axes[row, column].set_title(
            f"{test_name}\nInput - true:\n{wrapped(names[label])}", fontsize=7.8, weight="bold"
        )
        overlay_cam(axes[row, column + 1], tested, mask, baseline["cam"])
        axes[row, column + 1].set_title(
            f"CE - wrong:\n{wrapped(names[baseline['prediction']])}\n"
            f"confidence {100 * baseline['confidence']:.1f}%", fontsize=7.8,
        )
        overlay_cam(axes[row, column + 2], tested, mask, v2["cam"])
        axes[row, column + 2].set_title(
            f"FCAL V2 - correct\n{wrapped(names[label])}\n"
            f"confidence {100 * v2['confidence']:.1f}%",
            fontsize=7.8,
            weight="bold",
            color="#1F6B4A",
        )
        for axis in axes[row, column:column + 3]:
            axis.axis("off")

    color_scale = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet")
    colorbar = fig.colorbar(
        color_scale, ax=axes, orientation="horizontal", fraction=0.025, pad=0.018, aspect=55
    )
    colorbar.set_ticks([0, 1], labels=["Low evidence", "High evidence"])
    colorbar.set_label("Shared robust per-map normalization (2nd-98th percentile)")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.savefig(REPORT / output_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_figures() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    data = {
        architecture: [load_metrics(run_id) for run_id in run_ids]
        for architecture, run_ids in ARCHITECTURES.items()
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#2D3640",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), constrained_layout=True)
    x = np.arange(len(ARCHITECTURES))
    width = 0.24
    labels = list(ARCHITECTURES)
    panels = (
        ("Clean", "Clean top-1 accuracy (%)", True),
        ("BG mean", "Mean over 13 background tests (%)", True),
        ("BG worst", "Worst background accuracy (%)", True),
        ("FG-only", "Foreground-only accuracy (%)", True),
        ("Leakage", "Strict attention leakage (lower is better)", False),
    )

    for ax, (metric, title, higher_is_better) in zip(axes.flat, panels):
        all_values: list[float] = []
        for method_index, (method, color) in enumerate(zip(METHODS, COLORS)):
            values = [data[architecture][method_index][metric] for architecture in labels]
            all_values.extend(values)
            ax.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=method,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(title, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
        ax.set_axisbelow(True)
        if higher_is_better:
            lower = max(0.0, min(all_values) - 5.0)
            upper = min(100.0, max(all_values) + 3.0)
            ax.set_ylim(lower, upper)
        else:
            ax.set_ylim(0.0, 0.9)

    summary_ax = axes.flat[-1]
    summary_ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    summary_ax.legend(handles, METHODS, loc="upper left", frameon=False, ncol=3)
    summary_ax.text(
        0.0,
        0.72,
        "V2 leads all five metrics on\nResNet-50 and DenseNet-121.",
        fontsize=12,
        weight="bold",
        color="#1D2B36",
        va="top",
    )
    summary_ax.text(
        0.0,
        0.39,
        "Across all six architectures, V2 has\nthe lowest strict attention leakage.",
        fontsize=11,
        color="#1D2B36",
        va="top",
    )
    fig.suptitle(
        "FCAL V1 and V2 across six ImageNet-pretrained architectures on CUB-200-2011",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(REPORT / "multiarchitecture_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    rescue_family_figure(
        "ResNet-50 CNN",
        "resnet_50",
        "ResNet-50 CNN: CE wrong -> FCAL V2 correct across test families",
        "cnn_test_family_rescues.png",
    )
    rescue_family_figure(
        "ViT-B/16 average pool",
        "vit_base_16",
        "ViT-B/16: CE wrong -> FCAL V2 correct across test families",
        "vit_test_family_rescues.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tables-only", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    parser.add_argument("--fixed-results", type=Path, default=FIXED_RESULTS)
    parser.add_argument("--tuned-results", type=Path, default=TUNED_RESULTS)
    parser.add_argument("--table-output", type=Path, default=GENERATED_TABLES)
    args = parser.parse_args()
    if not args.figures_only:
        print(*(str(path) for path in build_comparison_tables(
            args.fixed_results, args.tuned_results, args.table_output,
        )), sep="\n")
    if not args.tables_only:
        build_figures()


if __name__ == "__main__":
    main()
