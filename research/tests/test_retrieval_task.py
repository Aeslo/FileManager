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
    # (query_text, relevant_doc_index)
    queries = [
        ("a dog is a great companion animal", 0),
        ("space exploration and rockets", 2),
    ]

    engine = TfidfEngine()
    engine.fit(corpus)

    task = RetrievalTask(k=2)
    metrics = task.run(engine, (corpus, queries))

    assert "precision_at_k" in metrics
    assert "mrr" in metrics
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_retrieval_task_perfect_score():
    """When the query is identical to a document it should rank first."""
    corpus = [
        "machine learning algorithms",
        "cooking recipes and food",
    ]
    queries = [
        ("machine learning algorithms", 0),
        ("cooking recipes and food", 1),
    ]

    engine = TfidfEngine()
    engine.fit(corpus)

    task = RetrievalTask(k=1)
    metrics = task.run(engine, (corpus, queries))

    assert metrics["precision_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_task_works_on_raw_vectors():
    """Task should accept pre-computed embeddings (modality-agnostic path)."""
    # Simulate image embeddings as random vectors
    rng = np.random.default_rng(42)
    corpus_vectors = rng.random((4, 8))
    # Query vector identical to doc 2
    query_vector = corpus_vectors[2].copy()

    task = RetrievalTask(k=1)
    metrics = task.run_from_vectors(corpus_vectors, [(query_vector, 2)])

    assert metrics["precision_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
