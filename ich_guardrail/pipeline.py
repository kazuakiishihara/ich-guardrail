from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ich_guardrail.masks import axial_slice_findings, intracranial_mask, reject_extracranial


def write_mask_like(mask_zyx: np.ndarray, ref: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path))


def segment_case(volume_path: Path, out_dir: Path, *, segmenter) -> dict:
    case = volume_path.name.replace(".nii.gz", "").replace(".nii", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Segmenting hematoma (HemSeg nnU-Net) for {case} ...")
    hematoma = segmenter.predict(str(volume_path))["hematoma"]

    ref = sitk.ReadImage(str(volume_path))
    spacing_xyz = tuple(float(s) for s in ref.GetSpacing())

    ct_zyx = sitk.GetArrayFromImage(ref)
    sp_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    intra = intracranial_mask(ct_zyx, sp_zyx, slice_axis=0)
    hematoma, dropped = reject_extracranial(hematoma, intra, sp_zyx)
    if dropped:
        print(f"[info] dropped {len(dropped)} extracranial focus/foci: "
              f"{', '.join(f'{v:.2f}mL' for v in dropped)}")

    hematoma_vox = int((hematoma > 0).sum())
    if hematoma_vox == 0:
        print("[warn] Empty hematoma mask; the payload will have no axial slices.")
    write_mask_like(hematoma, ref, out_dir / "hematoma.nii.gz")

    print(f"[done] {case}: segmented hematoma_vox={hematoma_vox} -> {out_dir}")
    return {"case": case, "hematoma_vox": hematoma_vox, "dropped": dropped}


def assemble_case(volume_path: Path, out_dir: Path) -> dict:
    case = volume_path.name.replace(".nii.gz", "").replace(".nii", "")
    findings = axial_slice_findings(out_dir / "hematoma.nii.gz")

    total_ml = round(sum(s["hematoma_volume"] for s in findings), 2)
    print(f"[done] {case}: hematoma spans {len(findings)} axial slices, total {total_ml} mL")
    return {"axial_slice": findings}
