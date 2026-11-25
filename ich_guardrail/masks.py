from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

CC_STRUCTURE = np.ones((3, 3, 3))
INTRACRANIAL_ERODE_MM = 8.0
REJECT_INSIDE_FRAC = 0.5


def _components(mask: np.ndarray, sp: np.ndarray) -> list[tuple[np.ndarray, float]]:
    lab, n = ndimage.label(mask, structure=CC_STRUCTURE)
    if n == 0:
        return []
    ml_per_vox = float(np.prod(sp)) / 1000.0
    vols = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) * ml_per_vox
    order = np.argsort(vols)[::-1]
    return [((lab == int(i + 1)), float(vols[i])) for i in order]


def intracranial_mask(ct: np.ndarray, sp: np.ndarray, slice_axis: int = 0) -> np.ndarray:
    soft = ndimage.binary_closing((ct >= -10) & (ct <= 100), iterations=2)
    se = np.ones((3, 3, 3), dtype=bool)
    se[(slice(None),) * slice_axis + (np.array([0, 2]),)] = False
    inplane = np.delete(np.asarray(sp, float), slice_axis)
    iters = max(1, int(round(INTRACRANIAL_ERODE_MM / float(inplane.min()))))
    core = ndimage.binary_erosion(soft, structure=se, iterations=iters)
    labels, n = ndimage.label(core)
    if n == 0:
        return np.zeros_like(ct, dtype=bool)
    largest = 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))))
    grown = ndimage.binary_dilation(labels == largest, structure=se, iterations=iters) & soft
    return ndimage.binary_fill_holes(grown)


def reject_extracranial(hematoma: np.ndarray, intra: np.ndarray, sp: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cleaned = np.zeros_like(hematoma, dtype=bool)
    dropped: list[float] = []
    for comp, vol in _components(hematoma > 0, np.asarray(sp, float)):
        if float(np.mean(intra[comp])) >= REJECT_INSIDE_FRAC:
            cleaned |= comp
        else:
            dropped.append(round(vol, 2))
    return cleaned, dropped


def _load_hematoma(hematoma_path: Path) -> tuple[np.ndarray, float]:
    img = nib.as_closest_canonical(nib.load(str(hematoma_path)))
    sp = np.asarray(img.header.get_zooms()[:3], dtype=float)
    return np.asarray(img.dataobj) > 0, float(np.prod(sp)) / 1000.0


def axial_slice_findings(hematoma_path: Path) -> list[dict]:
    hematoma, voxel_ml = _load_hematoma(hematoma_path)

    out: list[dict] = []
    for z in np.where(hematoma.any(axis=(0, 1)))[0]:
        sl = hematoma[:, :, z]
        cx, cy = ndimage.center_of_mass(sl)
        out.append({"slice_z": int(z), "x": int(round(cx)), "y": int(round(cy)),
                    "hematoma_volume": round(int(sl.sum()) * voxel_ml, 3)})
    return out


def hematoma_volume_ml(hematoma_path: Path) -> float:
    hematoma, voxel_ml = _load_hematoma(hematoma_path)
    return round(int(hematoma.sum()) * voxel_ml, 3)
