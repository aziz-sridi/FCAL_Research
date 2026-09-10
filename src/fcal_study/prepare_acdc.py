from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


LABELS = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}


@dataclass(frozen=True)
class Patient:
    patient_id: int
    directory: Path
    diagnosis: str
    ed_frame: int
    es_frame: int


def parse_info(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    return values


def discover_patients(source_root: Path) -> list[Patient]:
    patients = []
    for info_path in sorted(source_root.rglob("Info.cfg")):
        info = parse_info(info_path)
        diagnosis = info.get("group", "").upper()
        if diagnosis not in LABELS or "ed" not in info or "es" not in info:
            continue
        digits = "".join(character for character in info_path.parent.name if character.isdigit())
        if not digits:
            raise ValueError(f"Cannot infer patient number from {info_path.parent}")
        patients.append(Patient(
            patient_id=int(digits),
            directory=info_path.parent,
            diagnosis=diagnosis,
            ed_frame=int(info["ed"]),
            es_frame=int(info["es"]),
        ))
    if not patients:
        raise FileNotFoundError(
            f"No labeled ACDC patient directories were found below {source_root}. "
            "Expected patient*/Info.cfg with Group, ED, and ES fields."
        )
    duplicate_ids = len({patient.patient_id for patient in patients}) != len(patients)
    if duplicate_ids:
        raise ValueError("Duplicate labeled patient IDs found; point --source-root at one ACDC copy")
    return sorted(patients, key=lambda patient: patient.patient_id)


def stratified_patient_splits(
    patients: list[Patient], seed: int, validation_fraction: float, test_fraction: float,
) -> dict[str, list[Patient]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    rng = random.Random(seed)
    result = {"train": [], "validation": [], "test": []}
    for diagnosis in LABELS:
        members = [patient for patient in patients if patient.diagnosis == diagnosis]
        if len(members) < 3:
            raise ValueError(f"Need at least three {diagnosis} patients, found {len(members)}")
        rng.shuffle(members)
        test_count = max(1, round(len(members) * test_fraction))
        validation_count = max(1, round(len(members) * validation_fraction))
        if test_count + validation_count >= len(members):
            raise ValueError(f"Fractions leave no training patients for {diagnosis}")
        result["test"].extend(members[:test_count])
        result["validation"].extend(members[test_count:test_count + validation_count])
        result["train"].extend(members[test_count + validation_count:])
    for members in result.values():
        members.sort(key=lambda patient: patient.patient_id)
    return result


def _volume_paths(patient: Patient, frame: int) -> tuple[Path, Path]:
    stem = f"patient{patient.patient_id:03d}_frame{frame:02d}"
    for suffix in (".nii.gz", ".nii"):
        image_path = patient.directory / f"{stem}{suffix}"
        mask_path = patient.directory / f"{stem}_gt{suffix}"
        if image_path.is_file() and mask_path.is_file():
            return image_path, mask_path
    raise FileNotFoundError(
        f"Missing .nii[.gz] image/mask pair for {patient.directory / stem}"
    )


def _scale_mri(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    nonzero = finite[finite != 0]
    population = nonzero if nonzero.size else finite
    if not population.size:
        return np.zeros(volume.shape, dtype=np.uint8)
    low, high = np.percentile(population, [1.0, 99.0])
    if high <= low:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = np.clip((volume - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _load_phase(patient: Patient, frame: int) -> tuple[np.ndarray, np.ndarray]:
    image_path, mask_path = _volume_paths(patient, frame)
    image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError(f"Unexpected ACDC shapes for {patient.patient_id}: {image.shape}, {mask.shape}")
    return _scale_mri(image), mask > 0.5


def materialize_patient(
    patient: Patient, output_root: Path, minimum_foreground_pixels: int,
) -> list[dict[str, object]]:
    ed, ed_mask = _load_phase(patient, patient.ed_frame)
    es, es_mask = _load_phase(patient, patient.es_frame)
    if ed.shape != es.shape:
        raise ValueError(f"ED/ES shapes differ for patient {patient.patient_id}: {ed.shape}, {es.shape}")
    difference = np.abs(ed.astype(np.int16) - es.astype(np.int16)).astype(np.uint8)
    union = ed_mask | es_mask
    image_dir = output_root / "images" / f"patient{patient.patient_id:03d}"
    mask_dir = output_root / "masks" / f"patient{patient.patient_id:03d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for slice_index in range(union.shape[2]):
        foreground = union[:, :, slice_index]
        if int(foreground.sum()) < minimum_foreground_pixels:
            continue
        filename = f"slice{slice_index:02d}.png"
        image_path = image_dir / filename
        mask_path = mask_dir / filename
        channels = np.stack(
            [ed[:, :, slice_index], es[:, :, slice_index], difference[:, :, slice_index]], axis=-1,
        )
        Image.fromarray(channels, mode="RGB").save(image_path, optimize=True)
        Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").save(mask_path, optimize=True)
        records.append({
            "patient_id": patient.patient_id,
            "diagnosis": patient.diagnosis,
            "label": LABELS[patient.diagnosis],
            "slice_index": slice_index,
            "image_path": image_path.relative_to(output_root).as_posix(),
            "mask_path": mask_path.relative_to(output_root).as_posix(),
        })
    if not records:
        raise ValueError(f"No foreground slices retained for patient {patient.patient_id}")
    return records


def prepare(
    source_root: str | Path,
    output_root: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    minimum_foreground_pixels: int = 16,
) -> Path:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    patients = discover_patients(source_root)
    splits = stratified_patient_splits(
        patients, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction,
    )
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "numeric_id", "patient_id", "diagnosis", "label", "slice_index", "image_path", "mask_path",
    ]
    counts = {}
    numeric_id = 0
    for split, members in splits.items():
        rows = []
        for patient in members:
            for record in materialize_patient(patient, output_root, minimum_foreground_pixels):
                record["numeric_id"] = numeric_id
                numeric_id += 1
                rows.append(record)
        with (split_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = {
            "patients": len(members),
            "slices": len(rows),
            "patient_ids": [patient.patient_id for patient in members],
            "class_patients": {
                diagnosis: sum(patient.diagnosis == diagnosis for patient in members)
                for diagnosis in LABELS
            },
        }
    manifest = {
        "dataset": "ACDC MICCAI 2017",
        "source_root": str(source_root),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "minimum_foreground_pixels": minimum_foreground_pixels,
        "labels": LABELS,
        "input_channels": ["end_diastole", "end_systole", "absolute_difference"],
        "foreground": "union of ED and ES labels 1, 2, and 3",
        "splits": counts,
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return split_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare patient-grouped ACDC slices for FCAL")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-foreground-pixels", type=int, default=16)
    args = parser.parse_args()
    print(prepare(
        args.source_root,
        args.output_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        minimum_foreground_pixels=args.minimum_foreground_pixels,
    ))


if __name__ == "__main__":
    main()
