from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from multimedia.ports.extractor import FeatureExtractor

# Columnas de audio disponibles en el CSV de Spotify
_FEATURE_KEYS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


class AudioExtractor(FeatureExtractor):
    # Recibe la ruta al CSV con los features de audio
    def __init__(self, csv_path: str) -> None:
        self._data: dict[str, np.ndarray] = {}
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tid = row.get("track_id", "").strip()
                if not tid:
                    continue
                vector = np.array(
                    [float(row.get(k, 0.0) or 0.0) for k in _FEATURE_KEYS],
                    dtype=np.float32,
                )
                self._data[tid] = vector

    def extract(self, file_path: str) -> np.ndarray:
        # El nombre del archivo es el track_id
        track_id = Path(file_path).stem
        if track_id not in self._data:
            raise ValueError(f"Track no encontrado en CSV: {track_id}")
        return self._data[track_id].reshape(1, -1)

    def feature_dim(self) -> int:
        return len(_FEATURE_KEYS)

    def supported_formats(self) -> list[str]:
        return [""]
