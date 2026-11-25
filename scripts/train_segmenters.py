from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess

DATASET_ID = "502"
CONFIGURATION = "3d_fullres"

def _check_nnunet_env() -> None:
    missing = [
        v for v in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        if not os.environ.get(v)
    ]
    if missing:
        raise RuntimeError(
            f"Missing nnU-Net environment variables: {missing}."
        )


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _train_generic(dataset_id: str, args: argparse.Namespace) -> None:
    _check_nnunet_env()

    if not args.skip_preprocess:
        _run([
            "nnUNetv2_plan_and_preprocess",
            "-d", dataset_id,
            "--verify_dataset_integrity",
        ])

    if not args.skip_train:
        cmd = [
            "nnUNetv2_train",
            dataset_id,
            CONFIGURATION,
            args.fold,
            "-tr", args.trainer,
        ]
        if args.continue_training:
            cmd.append("--c")
        _run(cmd)


def _train_hemseg(args: argparse.Namespace) -> None:
    _check_nnunet_env()
    if not args.skip_convert:
        from training.convert_hemseg import run as convert_hemseg
        convert_hemseg(
            hemseg_root=args.hemseg_root,
            inclusion_csv=args.inclusion_csv,
            overwrite=args.overwrite_convert,
        )
    _train_generic(DATASET_ID, args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=["hemseg"], default="hemseg")
    p.add_argument("--hemseg-root", type=Path, default=Path("data/HemSeg-200"))
    p.add_argument("--inclusion-csv", type=Path, default=Path("info/hemseg_inclusion.csv"))
    p.add_argument("--fold", default="0")
    p.add_argument("--trainer",
        default="nnUNetTrainer_250epochs",
        help=("nnU-Net trainer class. Default 'nnUNetTrainer_250epochs' "
                "(~3-5h on RTX 4090). Use 'nnUNetTrainer' for full 1000 epochs.")
    )
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--overwrite-convert", action="store_true")
    p.add_argument("--continue-training", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _train_hemseg(args)


if __name__ == "__main__":
    main()
