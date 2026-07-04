from __future__ import annotations

import cv2
import numpy as np

from multimedia.ports.extractor import FeatureExtractor

# Tamaño máximo permitido antes de reducir la imagen
_MAX_DIM = 800


class SIFTExtractor(FeatureExtractor):
    # Número de descriptores SIFT a retener por imagen
    def __init__(self, n_keypoints: int = 200) -> None:
        self._sift = cv2.SIFT_create(nfeatures=n_keypoints)
        self._n_keypoints = n_keypoints

    def extract(self, file_path: str) -> np.ndarray:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {file_path}")
        img = self._resize_if_needed(img)
        _, descriptors = self._sift.detectAndCompute(img, None)
        if descriptors is None:
            # Imagen sin keypoints detectables
            return np.zeros((0, 128), dtype=np.float32)
        return descriptors.astype(np.float32)

    def feature_dim(self) -> int:
        return 128

    def supported_formats(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".bmp"]

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        # Reduce la imagen si alguna dimensión supera el límite
        h, w = img.shape[:2]
        if max(h, w) <= _MAX_DIM:
            return img
        scale = _MAX_DIM / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
