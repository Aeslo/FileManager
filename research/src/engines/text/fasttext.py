"""
FastText — Skip-gram with Negative Sampling + subword (character n-gram) vectors.

Reference: Bojanowski et al., 2017 (https://arxiv.org/abs/1607.04606)

Extends Word2Vec Skip-gram with one change: instead of a single vector per word,
each word is represented as the mean of its character n-gram vectors.

    Word2Vec:  center_vec = W[word]
    FastText:  center_vec = mean(W[g] for g in ngrams(word))

Gradients are distributed equally across all n-gram vectors of the center word.
OOV words at inference time are handled via their n-gram vectors alone.
"""

from collections import Counter

import numpy as np

from src.engines.base import BaseEngine
from src.engines.text.utils import sigmoid, tokenize


def _ngrams(word: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    """Character n-grams with FastText boundary markers: <word>."""
    padded = f"<{word}>"
    return [
        padded[i : i + n]
        for n in range(min_n, max_n + 1)
        for i in range(len(padded) - n + 1)
    ]


class FastTextEngine(BaseEngine):
    """FastText Skip-gram with negative sampling and subword vectors.

    Parameters
    ----------
    vector_size : int
        Dimensionality of subword vectors (default 100).
    window : int
        Max distance between center and context word (default 5).
    min_count : int
        Minimum word frequency to include in vocabulary (default 2).
    negative : int
        Number of negative samples per positive pair (default 5).
    epochs : int
        Number of passes over the corpus (default 5).
    lr : float
        Initial learning rate, linearly decayed to lr/100 (default 0.025).
    min_n, max_n : int
        Character n-gram size range (default 3–5).
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        negative: int = 5,
        epochs: int = 5,
        lr: float = 0.025,
        min_n: int = 3,
        max_n: int = 5,
        seed: int | None = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.min_n = min_n
        self.max_n = max_n
        self.seed = seed

        self._W: np.ndarray | None = None
        self._subword_vocab: list[str] = []
        self._subword2idx: dict[str, int] = {}
        self._word_ngrams: dict[str, list[int]] = {}
        self._word_vocab: list[str] = []
        self._noise_dist: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, corpus: list[str]) -> None:
        tokenized = [tokenize(doc) for doc in corpus]

        counts: Counter = Counter(w for doc in tokenized for w in doc)
        self._word_vocab = [w for w, c in counts.items() if c >= self.min_count]
        word_set = set(self._word_vocab)

        # Subword vocabulary: each word token + all its n-grams
        subword_tokens: list[str] = []
        for w in self._word_vocab:
            subword_tokens.append(w)
            subword_tokens.extend(_ngrams(w, self.min_n, self.max_n))
        self._subword_vocab = list(dict.fromkeys(subword_tokens))
        self._subword2idx = {t: i for i, t in enumerate(self._subword_vocab)}
        SV = len(self._subword_vocab)

        if SV == 0:
            raise ValueError("Subword vocabulary is empty after filtering.")

        # Pre-compute subword index lists per word (word token + its n-grams)
        self._word_ngrams = {
            w: [self._subword2idx[w]] + [
                self._subword2idx[g]
                for g in _ngrams(w, self.min_n, self.max_n)
                if g in self._subword2idx
            ]
            for w in self._word_vocab
        }

        freqs = np.array([counts[w] ** 0.75 for w in self._word_vocab], dtype=np.float64)
        self._noise_dist = freqs / freqs.sum()
        V = len(self._word_vocab)
        word2idx = {w: i for i, w in enumerate(self._word_vocab)}

        rng = np.random.default_rng(self.seed)
        self._W = rng.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, (SV, self.vector_size))
        C = np.zeros((V, self.vector_size), dtype=np.float64)

        total_words = sum(len(doc) for doc in tokenized)
        words_seen = 0
        lr = self.lr

        for epoch in range(self.epochs):
            for doc in tokenized:
                in_vocab = [(w, word2idx[w]) for w in doc if w in word_set]

                for pos, (center_word, center_word_idx) in enumerate(in_vocab):
                    actual_window = rng.integers(1, self.window + 1)
                    start = max(0, pos - actual_window)
                    end = min(len(in_vocab), pos + actual_window + 1)

                    ngram_idxs = self._word_ngrams[center_word]
                    center_vec = self._W[ngram_idxs].mean(axis=0)  # type: ignore[index]

                    for ctx_pos in range(start, end):
                        if ctx_pos == pos:
                            continue
                        ctx_word_idx = in_vocab[ctx_pos][1]

                        targets = [(ctx_word_idx, 1.0)]
                        for neg_idx in rng.choice(V, size=self.negative, p=self._noise_dist):
                            if neg_idx != ctx_word_idx:
                                targets.append((int(neg_idx), 0.0))

                        grad_center = np.zeros(self.vector_size)
                        for t_idx, label in targets:
                            error = sigmoid(np.dot(center_vec, C[t_idx])) - label
                            grad_center += error * C[t_idx]
                            C[t_idx] -= lr * error * center_vec

                        grad_per_ngram = grad_center / len(ngram_idxs)
                        for nidx in ngram_idxs:
                            self._W[nidx] -= lr * grad_per_ngram  # type: ignore[index]

                words_seen += len(doc)
                progress = (epoch * total_words + words_seen) / (self.epochs * total_words)
                lr = max(self.lr * (1.0 - progress), self.lr * 0.01)

        self.is_fitted = True

    def _word_vector(self, word: str) -> np.ndarray:
        if word in self._word_ngrams:
            idxs = self._word_ngrams[word]
        else:
            idxs = [self._subword2idx[g] for g in _ngrams(word, self.min_n, self.max_n) if g in self._subword2idx]
        if not idxs:
            return np.zeros(self.vector_size, dtype=np.float64)
        return self._W[idxs].mean(axis=0)  # type: ignore[index]

    def embed_text(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before embedding")
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float64)
        return np.mean([self._word_vector(t) for t in tokens], axis=0)
