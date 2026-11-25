from __future__ import annotations

import argparse
from pathlib import Path

from ich_guardrail.cohort import resolve_jobs
from ich_guardrail.pipeline import segment_case
from ich_guardrail.segmentation import HematomaSegmenter

DEFAULT_MODEL = "nnunet/results/Dataset502_HemSeg/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--volume", type=Path, help="Path to a single NCCT NIfTI.")
    src.add_argument("--cohort-csv", type=Path, help="CSV with a 'pid' column; batch over all pids.")
    p.add_argument("--nifti-dir", type=Path, default="data/nifti",
                   help="Directory of <pid:07d>.nii.gz for --cohort-csv.")
    p.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="nnU-Net folds to ensemble (default: all 5). Use a single fold to speed up.")
    p.add_argument("--out-root", type=Path, default="results/localize",
                   help="Per-case mask written to <out-root>/<case>/hematoma.nii.gz.")
    p.add_argument("--out-dir", type=Path, default=None, help="Single-case override of the output dir.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    jobs = resolve_jobs(volume=args.volume, cohort_csv=args.cohort_csv,
                        nifti_dir=args.nifti_dir, out_root=args.out_root, out_dir=args.out_dir)

    segmenter = HematomaSegmenter(
        model_folder=args.model_folder,
        device=args.device,
        use_folds=tuple(args.folds),
        checkpoint_name=args.checkpoint,
    )
    failed = []
    for i, (case_dir, volume_path) in enumerate(jobs, 1):
        print(f"\n===== [{i}/{len(jobs)}] {volume_path.name} =====")
        try:
            segment_case(volume_path, case_dir, segmenter=segmenter)
        except Exception as e:
            print(f"[error] {volume_path.name} failed: {e}")
            failed.append(volume_path.name)

    print(f"\n[summary] segmented {len(jobs) - len(failed)}/{len(jobs)} cases; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
