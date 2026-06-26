"""
MFCC: Mel-Frequency Cepstral Coefficients audio baseline.

Represents each audio clip as the mean + standard deviation of its MFCCs
pooled over time, giving a fixed-size vector regardless of clip length.
No learned features, analogous to ColorHistogram for images / TF-IDF for text.
Captures coarse timbre, but ignores temporal structure and semantics.
"""

import numpy as np
import librosa

from src.engines.base import BaseEngine


class MFCCEngine(BaseEngine):
    """Mel-frequency cepstral coefficient baseline engine.

    Parameters
    ---
    n_mfcc : Number of MFCC coefficients per frame (default 20). vector_size = n_mfcc * 2 (mean + std pooling over time)
    sample_rate : Target sample rate; audio is resampled on load (default 22050)
    """

    def __init__(self, n_mfcc: int = 20, sample_rate: int = 22050):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.vector_size = n_mfcc * 2

    def fit(self, corpus: list[str]) -> None:
        print("MFCCEngine is ready (no fitting required).")

    def embed_audio(self, audio_path: str) -> np.ndarray:
        try:
            y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception:
            return np.zeros(self.vector_size, dtype=np.float32)
        if y.size == 0:
            return np.zeros(self.vector_size, dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def embed_batch(self, inputs: list[str]) -> np.ndarray:
        return np.array([self.embed_audio(p) for p in inputs])
