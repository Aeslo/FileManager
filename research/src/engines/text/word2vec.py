"""
Word2Vec — Skip-gram with Negative Sampling (SGNS).

Reference: Mikolov et al., 2013 (https://arxiv.org/abs/1301.3781)

Two embedding matrices W (center) and C (context) are trained jointly.
For each (center, context) pair sampled within a sliding window:

    loss = log σ(W[c] · C[ctx]) + Σ log σ(−W[c] · C[neg])

W is kept as the word representation; C is discarded after training.
Document vectors are the mean of their constituent word vectors.
"""

from collections import Counter

import numpy as np

from src.engines.base import BaseEngine
from src.engines.text.utils import sigmoid, tokenize


class Word2VecEngine(BaseEngine):
    """Skip-gram Word2Vec with negative sampling, trained from scratch.

    Parameters
    ----------
    vector_size : int
        Dimensionality of word vectors (default 100).
    window : int
        Max distance between center and context word (default 5).
    min_count : int
        Ignore words that appear fewer than this many times (default 2).
    negative : int
        Number of negative samples per positive pair (default 5).
    epochs : int
        Number of passes over the corpus (default 5).
    lr : float
        Initial learning rate, linearly decayed to lr/100 (default 0.025).
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        negative: int = 5,
        epochs: int = 5,
        lr: float = 0.025,
        seed: int | None = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._W: np.ndarray | None = None
        self._vocab: list[str] = []
        self._word2idx: dict[str, int] = {}
        self._noise_dist: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, corpus: list[str]) -> None:
        tokenized = [tokenize(doc) for doc in corpus]

        counts: Counter = Counter(w for doc in tokenized for w in doc)
        self._vocab = [w for w, c in counts.items() if c >= self.min_count]
        self._word2idx = {w: i for i, w in enumerate(self._vocab)}
        V = len(self._vocab)

        if V == 0:
            raise ValueError("Vocabulary is empty after applying min_count filter.")

        # Noise distribution P(w) ∝ count(w)^(3/4) — Mikolov et al. §2.2
        freqs = np.array([counts[w] ** 0.75 for w in self._vocab], dtype=np.float64)
        self._noise_dist = freqs / freqs.sum()

        rng = np.random.default_rng(self.seed)
        self._W = rng.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, (V, self.vector_size))
        C = np.zeros((V, self.vector_size), dtype=np.float64)

        total_words = sum(len(doc) for doc in tokenized)
        words_seen = 0
        lr = self.lr

        for epoch in range(self.epochs):
            for doc in tokenized:
                indices = [self._word2idx[w] for w in doc if w in self._word2idx]

                for pos, center_idx in enumerate(indices):
                    actual_window = rng.integers(1, self.window + 1)
                    start = max(0, pos - actual_window)
                    end = min(len(indices), pos + actual_window + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == pos:
                            continue
                        ctx_idx = indices[ctx_pos]

                        targets = [(ctx_idx, 1.0)]
                        for neg_idx in rng.choice(V, size=self.negative, p=self._noise_dist):
                            if neg_idx != ctx_idx:
                                targets.append((int(neg_idx), 0.0))

                        w_center = self._W[center_idx].copy()
                        grad_center = np.zeros(self.vector_size)

                        for t_idx, label in targets:
                            error = sigmoid(np.dot(w_center, C[t_idx])) - label
                            grad_center += error * C[t_idx]
                            C[t_idx] -= lr * error * w_center

                        self._W[center_idx] -= lr * grad_center

                words_seen += len(doc)
                progress = (epoch * total_words + words_seen) / (self.epochs * total_words)
                lr = max(self.lr * (1.0 - progress), self.lr * 0.01)

        self.is_fitted = True

    def embed_text(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before embedding")
        tokens = tokenize(text)
        vecs = [self._W[self._word2idx[t]] for t in tokens if t in self._word2idx]  # type: ignore
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float64)
        return np.mean(vecs, axis=0)
