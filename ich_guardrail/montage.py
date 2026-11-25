from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

WL, WW = 40.0, 80.0


def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.as_closest_canonical(nib.load(str(path))).get_fdata())


def _window(ct: np.ndarray) -> np.ndarray:
    lo, hi = WL - WW / 2, WL + WW / 2
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def _pages(nz: int, per_page: int) -> list[range]:
    return [range(s, min(s + per_page, nz)) for s in range(0, max(nz, 1), per_page)]


def render_montage(ct_path: Path, out_dir: Path, rows: int = 5, cols: int = 5) -> list[Path]:
    ct_disp = np.rot90(_window(_load(ct_path)))
    nz = ct_disp.shape[2]

    out_dir = Path(out_dir)
    outs: list[Path] = []
    for page, zs in enumerate(_pages(nz, rows * cols), 1):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for k, z in enumerate(zs):
            ax = axes[k // cols][k % cols]
            ax.imshow(ct_disp[:, :, z], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.text(0.03, 0.97, f"z={z}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=14, color="yellow")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                            wspace=0.02, hspace=0.02)
        out = out_dir / f"montage_{page}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        outs.append(out)
    return outs
