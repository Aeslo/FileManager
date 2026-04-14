"""
SBERT — Sentence-BERT document embeddings.

Reference: Reimers & Gurevych, 2019 (https://arxiv.org/abs/1908.10084)

Unlike mean-pooled Word2Vec/FastText, SBERT produces a single fixed-size
embedding per sentence/document using a siamese fine-tuned transformer.
This makes it far better at capturing sentence-level semantics.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.engines.base import BaseEngine


class SBERTEngine(BaseEngine):
    """Sentence-BERT engine using a pretrained sentence-transformers model.

    Parameters
    ----------
    model_name : str
        Any model from https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
        Defaults to all-MiniLM-L6-v2 (fast, 384-dim).
    batch_size : int
        Batch size for encode() calls (default 64).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        print(f"Loading SBERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.encode("probe", convert_to_numpy=True).shape[0]
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"  Ready — embedding dim: {self.vector_size}")

    def fit(self, corpus: list[str]) -> None:
        """Pretrained model — no fitting required."""
        print(f"SBERTEngine ({self.model_name}) is ready (no fitting required).")

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)  # type: ignore[return-value]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(  # type: ignore[return-value]
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
