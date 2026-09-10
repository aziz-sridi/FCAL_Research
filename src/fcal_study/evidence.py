"""Publication-oriented statistics and efficiency accounting.

The functions in this module deliberately treat a training seed (not an image,
corruption, or epoch) as the experimental unit.  They are small enough to reuse
from report builders while also providing a command-line interface for audits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.integrate import trapezoid


def _columns(values: Sequence[str] | str) -> list[str]:
    return [values] if isinstance(values, str) else list(values)


def seed_summary(
    frame: pd.DataFrame,
    group_cols: Sequence[str] | str,
    metrics: Sequence[str] | str,
    *,
    seed_col: str = "seed",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Collapse repeated rows within seed, then form a t interval across seeds.

    Collapsing first prevents regional variants or repeated measurements from
    being incorrectly counted as independent training runs.  A one-seed group
    has an estimable mean but intentionally receives no interval.
    """
    groups, metric_names = _columns(group_cols), _columns(metrics)
    required = {*groups, seed_col, *metric_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    per_seed = frame.groupby([*groups, seed_col], dropna=False, sort=False)[metric_names].mean()
    rows: list[dict] = []
    level: str | list[str] = groups[0] if len(groups) == 1 else groups
    grouped = per_seed.groupby(level=level, dropna=False, sort=False)
    for key, values in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        base = dict(zip(groups, keys))
        for metric in metric_names:
            sample = values[metric].dropna().to_numpy(dtype=float)
            n = len(sample)
            mean = float(np.mean(sample)) if n else math.nan
            sd = float(np.std(sample, ddof=1)) if n > 1 else math.nan
            se = sd / math.sqrt(n) if n > 1 else math.nan
            half = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1) * se) if n > 1 else math.nan
            rows.append({
                **base,
                "metric": metric,
                "n_seeds": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "confidence": confidence,
                "ci_low": mean - half if n > 1 else math.nan,
                "ci_high": mean + half if n > 1 else math.nan,
            })
    return pd.DataFrame(rows)


def _sign_permutation_p(differences: np.ndarray) -> float:
    """Exact paired randomization p-value (two-sided) for at most 20 pairs."""
    observed = abs(float(np.mean(differences)))
    n = len(differences)
    if n == 0:
        return math.nan
    if n > 20:
        # Deterministic Monte Carlo keeps large comparisons inexpensive.
        rng = np.random.default_rng(0)
        signs = rng.choice((-1.0, 1.0), size=(100_000, n))
        null = np.abs(np.mean(signs * differences, axis=1))
        return float((1 + np.count_nonzero(null >= observed - 1e-15)) / (len(null) + 1))
    extreme = 0
    total = 1 << n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        value = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += value >= observed - 1e-15
    return extreme / total


def paired_comparison(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    metric: str,
    unit_cols: Sequence[str] | str,
    method_col: str = "method",
    confidence: float = 0.95,
    lower_is_better: bool = False,
    require_complete: bool = True,
) -> dict:
    """Compare methods on matched experimental units.

    ``mean_difference`` is always candidate minus reference in the metric's
    native units.  ``mean_improvement`` and effect sizes are sign-flipped for a
    lower-is-better metric, so positive values always favor the candidate.
    """
    units = _columns(unit_cols)
    required = {*units, method_col, metric}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    subset = frame[frame[method_col].isin((reference, candidate))]
    duplicate = subset.duplicated([*units, method_col], keep=False)
    if duplicate.any():
        raise ValueError("Each method must have one observation per paired unit")
    wide = subset.pivot(index=units, columns=method_col, values=metric)
    absent = [name for name in (reference, candidate) if name not in wide]
    if absent:
        raise ValueError(f"No observations for: {', '.join(absent)}")
    incomplete = int(wide[[reference, candidate]].isna().any(axis=1).sum())
    if incomplete and require_complete:
        raise ValueError(f"{incomplete} paired units are incomplete")
    paired = wide[[reference, candidate]].dropna()
    raw = paired[candidate].to_numpy(dtype=float) - paired[reference].to_numpy(dtype=float)
    improvement = -raw if lower_is_better else raw
    n = len(raw)
    mean_raw = float(np.mean(raw)) if n else math.nan
    mean_improvement = float(np.mean(improvement)) if n else math.nan
    sd = float(np.std(improvement, ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * se) if n > 1 else math.nan
    constant = n > 1 and np.allclose(improvement, improvement[0], rtol=1e-12, atol=1e-15)
    t_result = stats.ttest_1samp(improvement, 0.0) if n > 1 and not constant else None
    if constant and mean_improvement != 0:
        dz = math.copysign(math.inf, mean_improvement)
        paired_t_p = 0.0
    elif constant:
        dz = math.nan
        paired_t_p = 1.0
    else:
        dz = mean_improvement / sd if n > 1 and sd > 0 else math.nan
        paired_t_p = float(t_result.pvalue) if t_result is not None else math.nan
    # Small-sample correction for a standardized paired effect.
    hedges = dz * (1 - 3 / (4 * n - 5)) if n > 1 and math.isfinite(dz) else math.nan
    return {
        "reference": reference,
        "candidate": candidate,
        "metric": metric,
        "lower_is_better": lower_is_better,
        "n_pairs": n,
        "incomplete_pairs": incomplete,
        "mean_difference": mean_raw,
        "mean_improvement": mean_improvement,
        "sd_improvement": sd,
        "ci_low_improvement": mean_improvement - half if n > 1 else math.nan,
        "ci_high_improvement": mean_improvement + half if n > 1 else math.nan,
        "paired_t_p": paired_t_p,
        "sign_permutation_p": _sign_permutation_p(improvement),
        "cohens_dz": dz,
        "hedges_gz": hedges,
        "wins": int(np.count_nonzero(improvement > 0)),
        "ties": int(np.count_nonzero(improvement == 0)),
        "losses": int(np.count_nonzero(improvement < 0)),
    }


def paired_comparisons(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidates: Sequence[str],
    metrics: Sequence[str],
    unit_cols: Sequence[str] | str,
    method_col: str = "method",
    lower_is_better: Iterable[str] = (),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Build a tidy paired-inference table for several methods and metrics."""
    lower = set(lower_is_better)
    result = pd.DataFrame(
        paired_comparison(
            frame,
            reference=reference,
            candidate=candidate,
            metric=metric,
            unit_cols=unit_cols,
            method_col=method_col,
            confidence=confidence,
            lower_is_better=metric in lower,
        )
        for candidate in candidates
        for metric in metrics
    )
    for column in ("paired_t_p", "sign_permutation_p"):
        adjusted = np.full(len(result), np.nan)
        valid = np.flatnonzero(np.isfinite(result[column].to_numpy(dtype=float)))
        order = valid[np.argsort(result.iloc[valid][column].to_numpy(dtype=float))]
        running = 0.0
        for rank, index in enumerate(order):
            value = (len(order) - rank) * float(result.loc[index, column])
            running = max(running, value)
            adjusted[index] = min(1.0, running)
        result[f"{column}_holm"] = adjusted
    return result


def win_tie_loss(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    metrics: Sequence[str] | str,
    architecture_col: str = "architecture",
    method_col: str = "method",
    seed_col: str = "seed",
    lower_is_better: Iterable[str] = (),
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Count architecture-level wins after averaging matched training seeds."""
    metric_names = _columns(metrics)
    lower = set(lower_is_better)
    averaged = frame.groupby([architecture_col, method_col], sort=False)[metric_names].mean().reset_index()
    rows = []
    for metric in metric_names:
        wide = averaged.pivot(index=architecture_col, columns=method_col, values=metric)
        if reference not in wide or candidate not in wide:
            raise ValueError(f"Both methods are required for metric {metric}")
        complete = wide[[reference, candidate]].dropna()
        delta = complete[candidate] - complete[reference]
        improvement = -delta if metric in lower else delta
        rows.append({
            "metric": metric,
            "reference": reference,
            "candidate": candidate,
            "architectures": len(complete),
            "wins": int((improvement > tolerance).sum()),
            "ties": int((improvement.abs() <= tolerance).sum()),
            "losses": int((improvement < -tolerance).sum()),
            "tolerance": tolerance,
        })
    return pd.DataFrame(rows)


def pareto_front(
    frame: pd.DataFrame,
    *,
    maximize: Sequence[str] = (),
    minimize: Sequence[str] = (),
) -> pd.Series:
    """Return a Boolean mask for the non-dominated rows."""
    objectives = [*maximize, *minimize]
    if not objectives:
        raise ValueError("At least one Pareto objective is required")
    missing = sorted(set(objectives).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = frame[objectives].to_numpy(dtype=float)
    values[:, len(maximize):] *= -1.0
    valid = np.isfinite(values).all(axis=1)
    front = np.zeros(len(frame), dtype=bool)
    for index, point in enumerate(values):
        if not valid[index]:
            continue
        dominates = valid & np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        front[index] = not bool(np.any(dominates))
    return pd.Series(front, index=frame.index, name="pareto_optimal")


def attention_iou(
    attention: np.ndarray,
    masks: np.ndarray,
    *,
    threshold: float = 0.7,
    normalize_by_max: bool = True,
) -> np.ndarray:
    """Compute per-image thresholded attention/mask IoU.

    The default mirrors CAMAL's threshold of 0.7. Max normalization makes the
    threshold comparable across methods whose non-negative CAM scales differ.
    Empty unions are returned as NaN rather than being scored as perfect.
    """
    maps = np.asarray(attention, dtype=float)
    truth = np.asarray(masks, dtype=bool)
    if maps.shape != truth.shape or maps.ndim < 2:
        raise ValueError("attention and masks must have equal [N, ...] shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if normalize_by_max:
        flat = maps.reshape(len(maps), -1)
        maxima = np.max(flat, axis=1).reshape((len(maps),) + (1,) * (maps.ndim - 1))
        maps = np.divide(maps, maxima, out=np.zeros_like(maps), where=maxima > 0)
    predicted = maps > threshold
    axes = tuple(range(1, maps.ndim))
    intersection = np.logical_and(predicted, truth).sum(axis=axes)
    union = np.logical_or(predicted, truth).sum(axis=axes)
    return np.divide(
        intersection, union, out=np.full(len(maps), np.nan, dtype=float), where=union > 0,
    )


def ranked_perturbation_masks(attention: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    """Reveal the top-attention pixels at each fraction as ``[N, K, ...]`` masks."""
    maps = np.asarray(attention, dtype=float)
    if maps.ndim < 2:
        raise ValueError("attention must have shape [N, ...]")
    points = np.asarray(fractions, dtype=float)
    if points.ndim != 1 or len(points) < 2 or np.any(np.diff(points) < 0):
        raise ValueError("fractions must be a sorted one-dimensional sequence")
    if np.any((points < 0) | (points > 1)):
        raise ValueError("fractions must be in [0, 1]")
    flat = maps.reshape(len(maps), -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    result = np.zeros((len(maps), len(points), flat.shape[1]), dtype=bool)
    for point_index, fraction in enumerate(points):
        count = int(round(float(fraction) * flat.shape[1]))
        if count:
            np.put_along_axis(result[:, point_index], order[:, :count], True, axis=1)
    return result.reshape((len(maps), len(points), *maps.shape[1:]))


def perturbation_images(
    images: np.ndarray,
    attention: np.ndarray,
    fractions: Sequence[float],
    *,
    mode: str,
    baseline: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Construct CAM-ranked removal or insertion images for model scoring."""
    source = np.asarray(images)
    maps = np.asarray(attention)
    if source.ndim != maps.ndim + 1 or source.shape[0] != maps.shape[0] or source.shape[2:] != maps.shape[1:]:
        raise ValueError("images and attention shapes are incompatible")
    if mode not in {"removal", "insertion"}:
        raise ValueError("mode must be removal or insertion")
    ranked = ranked_perturbation_masks(maps, fractions)[:, :, None]
    base = np.asarray(baseline, dtype=source.dtype)
    expanded = source[:, None]
    if mode == "insertion":
        return np.where(ranked, expanded, base)
    return np.where(ranked, base, expanded)


def faithfulness_auc(fractions: Sequence[float], confidences: np.ndarray) -> np.ndarray:
    """Integrate per-image confidence curves over perturbed pixel fraction."""
    points = np.asarray(fractions, dtype=float)
    values = np.asarray(confidences, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(points):
        raise ValueError("confidences must have shape [N, len(fractions)]")
    if len(points) < 2 or np.any(np.diff(points) <= 0) or points[0] != 0 or points[-1] != 1:
        raise ValueError("fractions must increase strictly from 0 to 1")
    if np.any(~np.isfinite(values)):
        raise ValueError("confidences must be finite")
    return trapezoid(values, x=points, axis=1)


def _read_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_efficiency(run_dir: Path, *, branch_epochs: int | None = None) -> dict:
    """Aggregate timing, throughput, memory, and provenance for one run."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    history = _read_history(run_dir / "history.jsonl")
    if not history:
        raise ValueError(f"Empty history: {run_dir}")
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.is_file() else {}
    train_seconds = sum(float(row["train"]["seconds"]) for row in history)
    validation_seconds = sum(float(row["validation"]["seconds"]) for row in history)
    train_samples = sum(int(row["train"]["samples"]) for row in history)
    peak_bytes = max(int(row["train"].get("peak_cuda_memory_bytes", 0)) for row in history)
    # Branch accounting must be explicit: a warm-up length alone does not prove
    # that another run actually resumed the same checkpoint.
    if branch_epochs is None:
        branch_epochs = int(config.get("efficiency", {}).get("common_branch_epochs", 0))
    branch_epoch = int(branch_epochs)
    if branch_epoch < 0:
        raise ValueError("branch_epochs cannot be negative")
    branch_epoch = min(branch_epoch, len(history))
    post_branch = history[branch_epoch:]
    post_train_seconds = sum(float(row["train"]["seconds"]) for row in post_branch)
    post_validation_seconds = sum(float(row["validation"]["seconds"]) for row in post_branch)
    post_train_samples = sum(int(row["train"]["samples"]) for row in post_branch)
    family = config.get("model", {}).get("family")
    architecture = config.get("model", {}).get("name")
    return {
        "run_id": config.get("run_id", str(run_dir)),
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "method": config["method"],
        "family": family,
        "architecture": architecture,
        "epochs_recorded": len(history),
        "train_samples": train_samples,
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "wall_seconds": train_seconds + validation_seconds,
        "gpu_hours": (train_seconds + validation_seconds) / 3600.0 if environment.get("cuda_available") else 0.0,
        "train_samples_per_second": train_samples / train_seconds if train_seconds > 0 else math.nan,
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_gib": peak_bytes / 2**30,
        "common_branch_epochs": branch_epoch,
        "post_branch_train_samples": post_train_samples,
        "post_branch_train_seconds": post_train_seconds,
        "post_branch_validation_seconds": post_validation_seconds,
        "post_branch_wall_seconds": post_train_seconds + post_validation_seconds,
        "post_branch_gpu_hours": (post_train_seconds + post_validation_seconds) / 3600.0
        if environment.get("cuda_available") else 0.0,
        "post_branch_train_samples_per_second": post_train_samples / post_train_seconds
        if post_train_seconds > 0 else math.nan,
        "cuda_device": environment.get("cuda_device"),
        "config_hash": environment.get("config_hash"),
    }


def efficiency_overhead(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidates: Sequence[str],
    unit_cols: Sequence[str] = ("architecture", "seed"),
    method_col: str = "method",
) -> pd.DataFrame:
    """Return matched absolute and relative overhead for each candidate run."""
    units = list(unit_cols)
    costs = (
        "train_seconds", "validation_seconds", "wall_seconds", "gpu_hours",
        "peak_cuda_memory_gib", "post_branch_train_seconds",
        "post_branch_validation_seconds", "post_branch_wall_seconds", "post_branch_gpu_hours",
    )
    rates = ("train_samples_per_second", "post_branch_train_samples_per_second")
    required = {*units, method_col, *costs, *rates}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    duplicate = frame[frame[method_col].isin([reference, *candidates])].duplicated(
        [*units, method_col], keep=False,
    )
    if duplicate.any():
        raise ValueError("Each method must have one efficiency row per paired unit")
    rows: list[dict] = []
    indexed = frame.set_index([*units, method_col])
    for candidate in candidates:
        reference_rows = indexed.xs(reference, level=method_col).sort_index()
        candidate_rows = indexed.xs(candidate, level=method_col).sort_index()
        if not reference_rows.index.equals(candidate_rows.index):
            missing_units = reference_rows.index.symmetric_difference(candidate_rows.index)
            raise ValueError(f"Efficiency pairs are incomplete: {len(missing_units)} unmatched units")
        for unit, reference_row in reference_rows.iterrows():
            unit_key = unit if isinstance(unit, tuple) else (unit,)
            candidate_row = candidate_rows.loc[unit]
            row = dict(zip(units, unit_key))
            row.update({"reference": reference, "candidate": candidate})
            for metric in costs:
                base, value = float(reference_row[metric]), float(candidate_row[metric])
                row[f"{metric}_difference"] = value - base
                row[f"{metric}_ratio"] = value / base if base else math.nan
                row[f"{metric}_overhead_percent"] = 100.0 * (value / base - 1.0) if base else math.nan
            for metric in rates:
                base, value = float(reference_row[metric]), float(candidate_row[metric])
                row[f"{metric}_difference"] = value - base
                row[f"{metric}_ratio"] = value / base if base else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def tuning_audit(
    trials: pd.DataFrame,
    *,
    method_col: str = "method",
    split_col: str = "selection_split",
    selected_col: str = "selected",
    gpu_hours_col: str = "gpu_hours",
) -> pd.DataFrame:
    """Validate a sealed-test tuning manifest and total the search budget.

    Every trial must declare ``selection_split=validation``.  Exactly one trial
    per method may be selected.  If ``test_evaluated`` is supplied, only that
    frozen selection may have been evaluated on test.
    """
    required = {method_col, split_col, selected_col, gpu_hours_col}
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    normalized_split = trials[split_col].astype(str).str.strip().str.lower()
    if "comparison_regime" in trials:
        regime = trials["comparison_regime"].astype(str).str.strip().str.lower()
        allowed_regimes = {"fixed_default", "equal_tuned"}
        invalid_regimes = sorted(set(regime).difference(allowed_regimes))
        if invalid_regimes:
            raise ValueError(f"Invalid comparison regimes: {', '.join(invalid_regimes)}")
        tuned = regime.eq("equal_tuned")
        if not normalized_split[tuned].eq("validation").all():
            raise ValueError("All hyperparameter selection must use validation only")
        if not normalized_split[~tuned].isin(("fixed", "preregistered")).all():
            raise ValueError("Fixed defaults must be declared fixed or preregistered")
    else:
        regime = pd.Series("equal_tuned", index=trials.index)
        if not normalized_split.eq("validation").all():
            raise ValueError("All hyperparameter selection must use validation only")
    def booleans(column: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(column):
            return column
        normalized = column.astype(str).str.strip().str.lower()
        allowed = {"true", "false", "1", "0"}
        invalid = sorted(set(normalized).difference(allowed))
        if invalid:
            raise ValueError(f"Invalid Boolean values: {', '.join(invalid)}")
        return normalized.isin(("true", "1"))

    selected = booleans(trials[selected_col])
    group_keys = [regime, trials[method_col]]
    counts = selected.groupby(group_keys).sum()
    if not counts.eq(1).all():
        raise ValueError("Exactly one frozen selection is required per method")
    sizes = trials.groupby(group_keys).size()
    fixed_groups = sizes.loc["fixed_default"] if "fixed_default" in sizes.index.get_level_values(0) else pd.Series(dtype=int)
    if not fixed_groups.eq(1).all():
        raise ValueError("A fixed-default comparison must have one declared run per method")
    if "test_evaluated" in trials:
        tested = booleans(trials["test_evaluated"])
        if bool((tested & ~selected).any()):
            raise ValueError("A non-selected tuning trial was evaluated on test")
        tested_counts = tested.groupby(group_keys).sum()
        if not tested_counts.eq(1).all():
            raise ValueError("Exactly the frozen selection must be evaluated on test")
    result = trials.assign(_selected=selected, _regime=regime).groupby(
        ["_regime", method_col], sort=False,
    ).agg(
        tuning_trials=(method_col, "size"),
        total_gpu_hours=(gpu_hours_col, "sum"),
        selected_trial=("_selected", "sum"),
    ).reset_index()
    result = result.rename(columns={"_regime": "comparison_regime"})
    result["tuning_gpu_hours"] = np.where(
        result["comparison_regime"].eq("equal_tuned"), result["total_gpu_hours"], 0.0,
    )
    result["equal_trial_budget"] = result.groupby("comparison_regime")["tuning_trials"].transform(
        lambda values: values.eq(values.iloc[0])
    )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="FCAL statistical and efficiency audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    efficiency = subparsers.add_parser("efficiency")
    efficiency.add_argument("run_dirs", nargs="+", type=Path)
    efficiency.add_argument("--branch-epochs", type=int)
    efficiency.add_argument("--output", required=True, type=Path)
    pareto = subparsers.add_parser("pareto")
    pareto.add_argument("csv", type=Path)
    pareto.add_argument("--maximize", nargs="*", default=[])
    pareto.add_argument("--minimize", nargs="*", default=[])
    pareto.add_argument("--output", required=True, type=Path)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("csv", type=Path)
    summarize.add_argument("--groups", nargs="+", required=True)
    summarize.add_argument("--metrics", nargs="+", required=True)
    summarize.add_argument("--seed-column", default="seed")
    summarize.add_argument("--output", required=True, type=Path)
    paired = subparsers.add_parser("paired")
    paired.add_argument("csv", type=Path)
    paired.add_argument("--reference", required=True)
    paired.add_argument("--candidates", nargs="+", required=True)
    paired.add_argument("--metrics", nargs="+", required=True)
    paired.add_argument("--units", nargs="+", required=True)
    paired.add_argument("--lower-is-better", nargs="*", default=[])
    paired.add_argument("--output", required=True, type=Path)
    audit = subparsers.add_parser("tuning-audit")
    audit.add_argument("csv", type=Path)
    audit.add_argument("--output", required=True, type=Path)
    overhead = subparsers.add_parser("efficiency-overhead")
    overhead.add_argument("csv", type=Path)
    overhead.add_argument("--reference", required=True)
    overhead.add_argument("--candidates", nargs="+", required=True)
    overhead.add_argument("--units", nargs="+", default=["architecture", "seed"])
    overhead.add_argument("--output", required=True, type=Path)
    wtl = subparsers.add_parser("win-tie-loss")
    wtl.add_argument("csv", type=Path)
    wtl.add_argument("--reference", required=True)
    wtl.add_argument("--candidate", required=True)
    wtl.add_argument("--metrics", nargs="+", required=True)
    wtl.add_argument("--lower-is-better", nargs="*", default=[])
    wtl.add_argument("--tolerance", type=float, default=0.0)
    wtl.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "efficiency":
        output = pd.DataFrame(
            run_efficiency(path, branch_epochs=args.branch_epochs) for path in args.run_dirs
        )
    elif args.command == "pareto":
        output = pd.read_csv(args.csv)
        output["pareto_optimal"] = pareto_front(
            output, maximize=args.maximize, minimize=args.minimize,
        )
    elif args.command == "summarize":
        output = seed_summary(
            pd.read_csv(args.csv), args.groups, args.metrics, seed_col=args.seed_column,
        )
    elif args.command == "paired":
        output = paired_comparisons(
            pd.read_csv(args.csv), reference=args.reference,
            candidates=args.candidates, metrics=args.metrics, unit_cols=args.units,
            lower_is_better=args.lower_is_better,
        )
    elif args.command == "tuning-audit":
        output = tuning_audit(pd.read_csv(args.csv))
    elif args.command == "efficiency-overhead":
        output = efficiency_overhead(
            pd.read_csv(args.csv), reference=args.reference,
            candidates=args.candidates, unit_cols=args.units,
        )
    else:
        output = win_tie_loss(
            pd.read_csv(args.csv), reference=args.reference, candidate=args.candidate,
            metrics=args.metrics, lower_is_better=args.lower_is_better,
            tolerance=args.tolerance,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    _main()
