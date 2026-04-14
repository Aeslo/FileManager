"""Shared utilities for text embedding engines."""

import re

import numpy as np


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens."""
    return re.findall(r"[a-z]+", text.lower())


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
