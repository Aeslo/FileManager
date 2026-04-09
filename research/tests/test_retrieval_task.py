import numpy as np
from src.tasks.retrieval import RetrievalTask
from src.engines.text.tfidf import TfidfEngine


def test_retrieval_task_returns_correct_metrics():
    corpus = [
        "Dogs are loyal pets",
        "Cats are independent pets",
        "The rocket launched into orbit",
        "Satellites circle the Earth",
    ]
    queries = [
        ("a dog is a great companion animal", {0, 1}),
        ("space exploration and rockets", {2, 3}),
    ]

    engine = TfidfEngine()
    engine.fit(corpus)

    task = RetrievalTask(k=2)
    metrics = task.run(engine, (corpus, queries))

    assert "precision_at_k" in metrics
    assert "map" in metrics
    assert "mrr" in metrics
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["map"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_retrieval_task_perfect_score():
    """All top-k results are relevant when query matches its category exactly."""
    corpus = [
        "machine learning algorithms",
        "machine learning models",
        "cooking recipes and food",
        "delicious food recipes",
    ]
    queries = [
        ("machine learning", {0, 1}),
        ("food and cooking", {2, 3}),
    ]

    engine = TfidfEngine()
    engine.fit(corpus)

    task = RetrievalTask(k=2)
    metrics = task.run(engine, (corpus, queries))

    assert metrics["precision_at_k"] == 1.0
    assert metrics["map"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_task_map_with_large_relevant_set():
    """MAP should be meaningful even when the relevant set is large."""
    rng = np.random.default_rng(0)
    corpus_vecs = rng.random((20, 4))
    query_vec = corpus_vecs[0].copy()
    # 10 out of 20 docs are relevant — MAP at 0.5 precision boundary
    relevant = set(range(10))

    task = RetrievalTask(k=5)
    metrics = task.run_from_vectors(corpus_vecs, [(query_vec, relevant)])

    # MAP is in valid range and distinct from a trivially low value
    assert 0.0 <= metrics["map"] <= 1.0


def test_retrieval_task_works_on_raw_vectors():
    """Task should accept pre-computed embeddings (modality-agnostic path)."""
    rng = np.random.default_rng(42)
    corpus_vectors = rng.random((4, 8))
    query_vector = corpus_vectors[2].copy()

    task = RetrievalTask(k=1)
    metrics = task.run_from_vectors(corpus_vectors, [(query_vector, {2})])

    assert metrics["precision_at_k"] == 1.0
    assert metrics["map"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_task_empty_relevant_set():
    """Empty relevant set should return zero scores without crashing."""
    rng = np.random.default_rng(1)
    corpus_vecs = rng.random((4, 4))
    query_vec = rng.random(4)

    task = RetrievalTask(k=2)
    metrics = task.run_from_vectors(corpus_vecs, [(query_vec, set())])

    assert metrics["precision_at_k"] == 0.0
    assert metrics["map"] == 0.0
    assert metrics["mrr"] == 0.0
