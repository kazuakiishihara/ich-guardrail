from __future__ import annotations

from pathlib import Path


def _candidate_stems(pid: int | str, admission_date: str | None = None) -> list[str]:
    p7 = f"{int(pid):07d}"
    stems: list[str] = []
    if admission_date:
        ymd = str(admission_date).strip().replace("-", "")
        if ymd:
            stems.append(f"{p7}_{ymd}")
    stems.append(p7)
    return stems


def resolve_nifti(nifti_dir: Path, pid: int | str, admission_date: str | None = None) -> Path | None:
    for stem in _candidate_stems(pid, admission_date):
        path = Path(nifti_dir) / f"{stem}.nii.gz"
        if path.exists():
            return path
    return None


def resolve_case_dir(out_root: Path, pid: int | str, exam_date: str | None) -> Path | None:
    if not exam_date:
        return None
    for stem in _candidate_stems(pid, exam_date):
        case_dir = Path(out_root) / stem
        if (case_dir / "hematoma.nii.gz").exists():
            return case_dir
    return None


def case_name(nifti_path: Path | str) -> str:
    return Path(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")


def resolve_jobs(*, volume: Path | None, cohort_csv: Path | None, nifti_dir: Path,
                 out_root: Path, out_dir: Path | None = None) -> list[tuple[Path, Path]]:
    
    import csv

    if volume is not None:
        if not Path(volume).exists():
            raise FileNotFoundError(f"Input volume not found: {volume}")
        case = case_name(volume)
        return [(out_dir or (out_root / case), Path(volume))]

    with open(cohort_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "pid" not in rows[0]:
        raise ValueError(f"'pid' column not found in {cohort_csv}")

    jobs, missing = [], []
    for row in rows:
        pid = int(row["pid"])
        nifti = resolve_nifti(nifti_dir, pid, row.get("admission_date"))
        if nifti is not None:
            jobs.append((out_root / case_name(nifti), nifti))
        else:
            missing.append(pid)
    print(f"[info] Matched {len(jobs)}/{len(rows)} cohort pids to NIfTI files.")
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        print(f"[warn] Missing NIfTI for pids: {preview}{' ...' if len(missing) > 5 else ''}")
    if not jobs:
        raise FileNotFoundError("No cohort pids matched a NIfTI file under --nifti-dir.")
    return jobs
