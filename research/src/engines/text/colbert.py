"""
ColBERT — Contextualized Late Interaction over BERT.

Reference: Khattab & Zaharia, 2020 (https://arxiv.org/abs/2004.12832)
           ColBERTv2: Santhanam et al., 2021 (https://arxiv.org/abs/2112.01488)

Key difference from SBERT:
  SBERT:   embed(doc) → single vector  →  score = q · d
  ColBERT: embed(doc) → token matrix   →  score = Σ_i max_j(q_i · d_j)   (MaxSim)

This late interaction lets the model retain fine-grained token-level signals
that mean-pooling destroys, while still being efficient at retrieval time
(document token matrices are pre-computed and indexed offline).

embed_text / embed_batch return mean-pooled token vectors so the engine
remains compatible with SortingTask. Retrieval must go through
encode_tokens_batch + MaxSim scoring (see ColBERTRetrievalTask).
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.engines.base import BaseEngine


class ColBERTEngine(BaseEngine):
    """ColBERT late-interaction engine using colbert-ir/colbertv2.0.

    Parameters
    ----------
    model_name : str
        HuggingFace model id. Defaults to the official ColBERTv2 checkpoint.
    max_length : int
        Max token length per document (default 180, same as ColBERT paper).
    batch_size : int
        Number of documents encoded per forward pass (default 32).
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_length: int = 180,
        batch_size: int = 32,
    ):
        print(f"Loading ColBERT model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        # Infer vector size from a probe forward pass
        self.vector_size = self._encode_tokens("probe").shape[1]
        print(f"  Ready — token embedding dim: {self.vector_size}")

    def fit(self, corpus: list[str]) -> None:
        """Pretrained model — no fitting required."""
        print(f"ColBERTEngine ({self.model_name}) is ready (no fitting required).")

    # ── Token-level encoding ──────────────────────────────────────────────────

    def _encode_tokens(self, text: str) -> np.ndarray:
        """Encode a single text into L2-normalised per-token vectors.

        Returns
        -------
        np.ndarray of shape (seq_len, hidden_size)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # (seq_len, hidden_size) — drop the batch dim
        token_vecs = outputs.last_hidden_state[0].cpu().numpy()
        # L2-normalise each token vector (required for MaxSim dot products)
        norms = np.linalg.norm(token_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return token_vecs / norms

    def encode_tokens_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode a list of texts into per-token vector matrices.

        Returns
        -------
        list of np.ndarray, each shaped (seq_len_i, hidden_size)
        Used by ColBERTRetrievalTask for MaxSim scoring.
        """
        return [self._encode_tokens(t) for t in texts]

    # ── Single-vector interface (for SortingTask compatibility) ───────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Mean pool token vectors → single document vector."""
        return self._encode_tokens(text).mean(axis=0)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts])
