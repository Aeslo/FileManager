import numpy as np
from src.tasks.base import BaseTask
from src.engines.text.colbert import ColBERTEngine


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _rank(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> list[int]:
    scores = [_cosine_similarity(query_vec, doc_vec) for doc_vec in corpus_vecs]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for idx in ranked[:k] if idx in relevant)
    return hits / k


def _average_precision(ranked: list[int], relevant: set[int]) -> float:
    """Averages precision at each position where a relevant doc appears.
    Unlike Precision@k, it accounts for the full ranking and the size of the relevant set."""
    if not relevant:
        return 0.0
    hits, precision_sum = 0, 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def _reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for rank, idx in enumerate(ranked, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def _maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT MaxSim: for each query token, find its best matching doc token.

    score(q, d) = Σ_i  max_j ( q_i · d_j )

    Both matrices must already be L2-normalised (as ColBERTEngine guarantees).
    """
    # (q_len, d_len) similarity matrix
    sim = query_tokens @ doc_tokens.T
    return float(sim.max(axis=1).sum())


class ColBERTRetrievalTask(BaseTask):
    """Retrieval task that uses ColBERT MaxSim scoring instead of cosine similarity.

    Expects a ColBERTEngine so it can call encode_tokens_batch().
    """

    def __init__(self, k: int = 5):
        self.k = k

    def run(self, engine: ColBERTEngine, dataset: tuple) -> dict[str, float]:
        corpus, queries = dataset
        print("  Encoding corpus token matrices (ColBERT)...")
        corpus_token_matrices = engine.encode_tokens_batch(corpus)

        p_scores, ap_scores, rr_scores = [], [], []
        for q_text, relevant in queries:
            q_tokens = engine._encode_tokens(q_text)
            scores = [_maxsim_score(q_tokens, d_tokens) for d_tokens in corpus_token_matrices]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            p_scores.append(_precision_at_k(ranked, relevant, self.k))
            ap_scores.append(_average_precision(ranked, relevant))
            rr_scores.append(_reciprocal_rank(ranked, relevant))

        return {
            "precision_at_k": float(np.mean(p_scores)),
            "map": float(np.mean(ap_scores)),
            "mrr": float(np.mean(rr_scores)),
        }


class RetrievalTask(BaseTask):
    """
    Given a query and a corpus of embeddings, ranks the corpus by cosine similarity
    and measures how well the relevant documents are surfaced.

    Relevant documents are defined as a set of corpus indices that are considered
    correct answers for a given query — e.g. all documents sharing the same category label.

    Queries must come from outside the corpus (e.g. a held-out test split) —
    a query identical to a corpus document will always rank itself first.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def run(self, engine, dataset: tuple[list[str], list[tuple[str, set[int]]]]) -> dict[str, float]:
        """
        dataset: (corpus, queries)
          corpus:  list[str]
          queries: list[tuple[str, set[int]]]  — (query_text, relevant_indices)
        """
        corpus, queries = dataset
        corpus_vecs = engine.embed_batch(corpus)
        vec_queries = [(engine.embed_text(q), rel) for q, rel in queries]
        return self.run_from_vectors(corpus_vecs, vec_queries)

    def run_from_vectors(
        self,
        corpus_vecs: np.ndarray,
        queries: list[tuple[np.ndarray, set[int]]],
    ) -> dict[str, float]:
        """
        Modality-agnostic path: accepts pre-computed vectors directly.
        queries: List of (query_vector, set_of_relevant_indices).
        """
        p_scores, ap_scores, rr_scores = [], [], []
        for query_vec, relevant in queries:
            ranked = _rank(query_vec, corpus_vecs)
            p_scores.append(_precision_at_k(ranked, relevant, self.k))
            ap_scores.append(_average_precision(ranked, relevant))
            rr_scores.append(_reciprocal_rank(ranked, relevant))

        return {
            "precision_at_k": float(np.mean(p_scores)),
            "map": float(np.mean(ap_scores)),
            "mrr": float(np.mean(rr_scores)),
        }
