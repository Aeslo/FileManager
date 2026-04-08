import numpy as np
from typing import Any, List, Tuple
from src.tasks.base import BaseTask


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _precision_at_k(ranked_indices: List[int], relevant_idx: int, k: int) -> float:
    return 1.0 if relevant_idx in ranked_indices[:k] else 0.0


def _reciprocal_rank(ranked_indices: List[int], relevant_idx: int) -> float:
    if relevant_idx in ranked_indices:
        return 1.0 / (ranked_indices.index(relevant_idx) + 1)
    return 0.0


def _rank(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> List[int]:
    scores = [_cosine_similarity(query_vec, doc_vec) for doc_vec in corpus_vecs]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class RetrievalTask(BaseTask):
    """
    Evaluates a vector-based retrieval system using Precision@k and Mean Reciprocal Rank (MRR).

    This task measures how effectively an engine ranks relevant documents within a corpus
    based on query similarities. It supports both raw text input (via an engine) and
    pre-computed vectors for modality-agnostic evaluation.
    """
    def __init__(self, k: int = 5):
        self.k = k

    def run(self, engine, dataset: Any) -> dict:
        """
        dataset: (corpus: List[str], queries: List[Tuple[str, int]])
        Each query is (query_text, relevant_doc_index).
        """
        corpus, queries = dataset
        corpus_vecs = np.array([engine.embed_text(doc) for doc in corpus])
        query_vecs = [(engine.embed_text(q), rel) for q, rel in queries]
        return self.run_from_vectors(corpus_vecs, query_vecs)

    def run_from_vectors(
        self,
        corpus_vecs: np.ndarray,
        queries: List[Tuple[np.ndarray, int]],
    ) -> dict:
        """
        Modality-agnostic path: accepts pre-computed vectors directly.
        queries: List of (query_vector, relevant_doc_index).
        """
        p_scores, rr_scores = [], []
        for query_vec, relevant_idx in queries:
            ranked = _rank(query_vec, corpus_vecs)
            p_scores.append(_precision_at_k(ranked, relevant_idx, self.k))
            rr_scores.append(_reciprocal_rank(ranked, relevant_idx))

        return {
            "precision_at_k": float(np.mean(p_scores)),
            "mrr": float(np.mean(rr_scores)),
        }
