from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import nibabel as nib
import numpy as np
import os
from pathlib import Path
from typing import Optional

import pandas as pd


DATASET_ID = 502
DATASET_NAME = f"Dataset{DATASET_ID}_HemSeg"

HEMSEG_LABELS: dict[str, int] = {
    "background": 0,
    "hemorrhage": 1,
}

DEFAULT_HEMSEG_ROOT = Path("data/HemSeg-200")
DEFAULT_INCLUSION_CSV = Path("info/hemseg_inclusion.csv")


def _nnunet_raw_root() -> Path:
    raw = os.environ.get("nnUNet_raw")
    if not raw:
        raise RuntimeError(
            "Environment variable nnUNet_raw is not set. Export it before running."
        )
    return Path(raw)


def _included_pids(inclusion_csv: Path) -> list[str]:
    df = pd.read_csv(inclusion_csv)
    for col in ("pid", "inclusion"):
        if col not in df.columns:
            raise ValueError(f"{inclusion_csv} is missing required column '{col}'.")
    included = df.loc[df["inclusion"].astype(bool), "pid"].astype(str)
    return sorted(included.tolist())


def _list_cases(hemseg_root: Path, pids: list[str]) -> list[str]:
    paired, missing = [], []
    for pid in pids:
        img = hemseg_root / "IMG" / f"{pid}.nii.gz"
        lab = hemseg_root / "IPH_IVH_label" / f"{pid}.nii.gz"
        (paired if img.exists() and lab.exists() else missing).append(pid)
    if missing:
        print(f"[warn] {len(missing)} included pids have no image/label pair; skipping: {missing[:5]}...")
    return paired


def _convert_one(
    src_image: Path,
    src_label: Path,
    dst_image: Path,
    dst_label: Path,
) -> Counter:
    img = nib.load(str(src_image))
    img_data = np.asarray(img.dataobj).astype(np.int16)
    nib.save(nib.Nifti1Image(img_data, img.affine, img.header), str(dst_image))

    lab = nib.load(str(src_label))
    lab_data = (np.asarray(lab.dataobj) > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(lab_data, lab.affine, lab.header), str(dst_label))

    values, counts = np.unique(lab_data, return_counts=True)
    return Counter(dict(zip(values.tolist(), counts.tolist())))


def _write_dataset_json(target_dir: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": HEMSEG_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "HemSeg-200: hemorrhage segmentation (IPH + IVH).",
    }
    (target_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))


def _write_case_mapping(target_dir: Path, mapping: list[tuple[str, str]]) -> None:
    with (target_dir / "case_mapping.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "pid"])
        writer.writerows(mapping)


def run(
    hemseg_root: Path = DEFAULT_HEMSEG_ROOT,
    inclusion_csv: Path = DEFAULT_INCLUSION_CSV,
    target_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:

    hemseg_root = Path(hemseg_root)
    inclusion_csv = Path(inclusion_csv)
    target_root = Path(target_root) if target_root else _nnunet_raw_root()
    dataset_dir = target_root / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    pids = _list_cases(hemseg_root, _included_pids(inclusion_csv))
    print(f"[info] Found {len(pids)} included cases with image/label pairs in {hemseg_root}.")

    if dry_run:
        print(f"[dry-run] Would write {len(pids)} cases to {dataset_dir}.")
        return dataset_dir

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    label_totals: Counter = Counter()
    mapping: list[tuple[str, str]] = []
    skipped = 0
    for idx, pid in enumerate(pids):
        case_id = f"HemSeg_{idx:03d}"
        mapping.append((case_id, pid))
        dst_image = images_tr / f"{case_id}_0000.nii.gz"
        dst_label = labels_tr / f"{case_id}.nii.gz"
        if dst_image.exists() and dst_label.exists() and not overwrite:
            skipped += 1
            continue
        per_case = _convert_one(
            src_image=hemseg_root / "IMG" / f"{pid}.nii.gz",
            src_label=hemseg_root / "IPH_IVH_label" / f"{pid}.nii.gz",
            dst_image=dst_image,
            dst_label=dst_label,
        )
        label_totals.update(per_case)
        if (idx + 1) % 25 == 0:
            print(f"[info] Converted {idx + 1}/{len(pids)}.")

    _write_dataset_json(dataset_dir, num_training=len(pids))
    _write_case_mapping(dataset_dir, mapping)

    print(f"[done] Wrote {len(pids) - skipped} new cases (skipped {skipped} existing).")
    print(f"[done] Dataset dir: {dataset_dir}")
    if label_totals:
        inv = {v: k for k, v in HEMSEG_LABELS.items()}
        print("[label-distribution] (voxels across all converted cases)")
        for label_value in sorted(label_totals):
            print(f"  {label_value} ({inv.get(label_value, '?')}): {label_totals[label_value]:>14,}")
        expected = set(HEMSEG_LABELS.values())
        missing_classes = expected - set(label_totals)
        if missing_classes:
            inv_names = [inv[v] for v in sorted(missing_classes)]
            print(f"[warn] Labels not observed in any case: {inv_names}")

    return dataset_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hemseg-root",
        type=Path,
        default=DEFAULT_HEMSEG_ROOT,
        help=f"Path to HemSeg-200 root (default: {DEFAULT_HEMSEG_ROOT}).",
    )
    p.add_argument(
        "--inclusion-csv",
        type=Path,
        default=DEFAULT_INCLUSION_CSV,
        help=f"Inclusion CSV with pid/inclusion columns (default: {DEFAULT_INCLUSION_CSV}).",
    )
    p.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Override target root (default: $nnUNet_raw).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        hemseg_root=args.hemseg_root,
        inclusion_csv=args.inclusion_csv,
        target_root=args.target_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
