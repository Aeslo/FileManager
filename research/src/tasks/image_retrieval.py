"""
Image-to-Image Retrieval Task.

Identical to RetrievalTask but uses embed_image() for query encoding
instead of embed_text(), since queries are image file paths.
"""

import numpy as np

from src.tasks.retrieval import (
    RetrievalTask,
    _precision_at_k,
    _average_precision,
    _reciprocal_rank,
    _rank,
)


class ImageRetrievalTask(RetrievalTask):
    """Retrieval task where both query and corpus are image file paths.

    dataset: (corpus_paths, queries)
      corpus_paths : list[str]
      queries      : list[tuple[str, set[int]]]  — (query_image_path, relevant_indices)
    """

    def run(self, engine, dataset: tuple[list[str], list[tuple[str, set[int]]]]) -> dict[str, float]:
        corpus_paths, queries = dataset
        corpus_vecs = engine.embed_batch(corpus_paths)

        p_scores, ap_scores, rr_scores = [], [], []
        for query_path, relevant in queries:
            query_vec = engine.embed_image(query_path)
            ranked = _rank(query_vec, corpus_vecs)
            p_scores.append(_precision_at_k(ranked, relevant, self.k))
            ap_scores.append(_average_precision(ranked, relevant))
            rr_scores.append(_reciprocal_rank(ranked, relevant))

        return {
            "precision_at_k": float(np.mean(p_scores)),
            "map": float(np.mean(ap_scores)),
            "mrr": float(np.mean(rr_scores)),
        }
