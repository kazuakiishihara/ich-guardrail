from __future__ import annotations

import argparse
import json
from pathlib import Path

from ich_guardrail.cohort import resolve_jobs
from ich_guardrail.montage import render_montage
from ich_guardrail.pipeline import assemble_case


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--volume", type=Path, help="Path to a single NCCT NIfTI.")
    src.add_argument("--cohort-csv", type=Path, help="CSV with a 'pid' column; batch over all pids.")
    p.add_argument("--nifti-dir", type=Path, default="data/nifti",
                   help="Directory of <pid:07d>.nii.gz for --cohort-csv.")
    p.add_argument("--out-root", type=Path, default="results/localize",
                   help="Per-case mask read from / outputs written to <out-root>/<case>/.")
    p.add_argument("--out-dir", type=Path, default=None, help="Single-case override of the output dir.")
    p.add_argument("--montage-grid", type=int, nargs=2, metavar=("ROWS", "COLS"), default=[5, 5],
                   help="Montage tile grid per page (default 5 5 = 25 slices/page); extra slices "
                        "spill onto montage_2.png, montage_3.png, ...")
    args = p.parse_args()

    jobs = resolve_jobs(volume=args.volume, cohort_csv=args.cohort_csv,
                        nifti_dir=args.nifti_dir, out_root=args.out_root, out_dir=args.out_dir)
    rows, cols = args.montage_grid

    failed, no_masks = [], []
    for i, (case_dir, volume_path) in enumerate(jobs, 1):
        print(f"\n===== [{i}/{len(jobs)}] {volume_path.name} =====")
        if not (case_dir / "hematoma.nii.gz").exists():
            print(f"[warn] {case_dir.name}: no mask; run scripts/segment.py first -- skipping.")
            no_masks.append(volume_path.name)
            continue
        try:
            payload = assemble_case(volume_path, case_dir)
            (case_dir / "payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            monts = render_montage(volume_path, case_dir, rows, cols)
            print(f"[done] {case_dir.name}: montage -> {', '.join(m.name for m in monts)}")
        except Exception as e:
            print(f"[error] {volume_path.name} failed: {e}")
            failed.append(volume_path.name)

    if len(no_masks) == len(jobs):
        raise FileNotFoundError("No case had a cached mask; run scripts/segment.py first.")
    done = len(jobs) - len(failed) - len(no_masks)
    print(f"\n[summary] assembled {done}/{len(jobs)} cases; "
          f"missing masks: {no_masks or 'none'}; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
