"""
Color Histogram — pixel-level image baseline.

Represents each image as a concatenation of per-channel RGB histograms.
No learned features — analogous to TF-IDF for text.
Fast to compute, works without a GPU, but completely ignores semantics.
"""

import numpy as np
import cv2

from src.engines.base import BaseEngine


class ColorHistogramEngine(BaseEngine):
    """RGB color histogram baseline engine.

    Parameters
    ----------
    bins : int
        Number of histogram bins per channel (default 32).
        vector_size = bins * 3.
    """

    def __init__(self, bins: int = 32):
        self.bins = bins
        self.vector_size = bins * 3

    def fit(self, corpus: list[str]) -> None:
        print(f"ColorHistogramEngine is ready (no fitting required).")

    def embed_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(self.vector_size, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = np.concatenate([
            np.histogram(img[:, :, c], bins=self.bins, range=(0, 256), density=True)[0]
            for c in range(3)
        ]).astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def embed_batch(self, inputs: list[str]) -> np.ndarray:
        return np.array([self.embed_image(p) for p in inputs])
