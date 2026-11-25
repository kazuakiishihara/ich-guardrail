from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable

_PREDICTOR_CACHE: dict[tuple, object] = {}


class HematomaSegmenter:
    LABEL_MAP: dict[str, int] = {"hematoma": 1}

    def __init__(
        self,
        model_folder: Path,
        device: str = "cuda",
        use_folds: Iterable[int] = (0,),
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        self.model_folder = Path(model_folder)
        self.device = device
        self.use_folds = tuple(use_folds)
        self.checkpoint_name = checkpoint_name
        self._predictor = None

    @property
    def _cache_key(self) -> tuple:
        return (str(self.model_folder), self.use_folds, self.device, self.checkpoint_name)

    def load(self) -> None:
        if self._predictor is not None:
            return
        cached = _PREDICTOR_CACHE.get(self._cache_key)
        if cached is not None:
            self._predictor = cached
            return

        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        if not self.model_folder.exists():
            raise FileNotFoundError(
                f"nnU-Net model folder not found: {self.model_folder}. "
                "Train the model first via scripts/train_segmenters.py --target hemseg."
            )

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(self.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(self.model_folder),
            use_folds=self.use_folds,
            checkpoint_name=self.checkpoint_name,
        )
        _PREDICTOR_CACHE[self._cache_key] = predictor
        self._predictor = predictor

    def predict(self, volume_path: str) -> dict[str, np.ndarray]:
        if self._predictor is None:
            self.load()

        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

        image, props = SimpleITKIO().read_images([volume_path])
        seg = self._predictor.predict_single_npy_array(
            input_image=image,
            image_properties=props,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        return {
            name: (seg == label).astype(np.uint8)
            for name, label in self.LABEL_MAP.items()
        }
