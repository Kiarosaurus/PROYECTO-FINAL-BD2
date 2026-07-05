from __future__ import annotations

import librosa
import numpy as np

from multimedia.ports.extractor import FeatureExtractor


class MFCCExtractor(FeatureExtractor):
    # Cantidad de coeficientes y tamaño de ventana configurables
    def __init__(
        self,
        n_mfcc: int = 20,
        window_frames: int = 32,
        hop_frames: int = 16,
        sample_rate: int = 22050,
    ) -> None:
        if window_frames <= 0 or hop_frames <= 0:
            raise ValueError("window_frames y hop_frames deben ser positivos")
        self._n_mfcc = n_mfcc
        self._window_frames = window_frames
        self._hop_frames = hop_frames
        self._sample_rate = sample_rate

    def extract(self, file_path: str) -> np.ndarray:
        try:
            signal, sr = librosa.load(file_path, sr=self._sample_rate, mono=True)
        except Exception as exc:
            raise ValueError(f"No se pudo leer el audio: {file_path}") from exc
        if signal.size == 0:
            # Audio sin muestras utilizables
            return np.zeros((0, self.feature_dim()), dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self._n_mfcc)
        frames = mfcc.T.astype(np.float32)
        return self._window_descriptors(frames)

    def feature_dim(self) -> int:
        # Media y desviación por cada coeficiente
        return 2 * self._n_mfcc

    def supported_formats(self) -> list[str]:
        return [".wav", ".mp3", ".ogg"]

    def _window_descriptors(self, frames: np.ndarray) -> np.ndarray:
        n_frames = frames.shape[0]
        # Audio más corto que una ventana usa una sola ventana con lo disponible
        if n_frames <= self._window_frames:
            starts: range | list[int] = [0]
            width = n_frames
        else:
            starts = range(0, n_frames - self._window_frames + 1, self._hop_frames)
            width = self._window_frames
        descriptors = []
        for start in starts:
            window = frames[start:start + width]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            descriptors.append(np.concatenate([mean, std]))
        return np.stack(descriptors).astype(np.float32)
