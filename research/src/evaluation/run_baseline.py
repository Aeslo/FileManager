"""
Baseline evaluation script.

Runs TF-IDF against both evaluation tasks on the 20 Newsgroups dataset,
prints results, and appends a dated entry to results/log.json.

Usage:
    python -m src.evaluation.run_baseline
"""

import json
import os
from collections import defaultdict
from datetime import date

from src.data_utils.loader import load_20newsgroups
from src.engines.text.tfidf import TfidfEngine
from src.tasks.sorting import SortingTask
from src.tasks.retrieval import RetrievalTask

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORIES = [
    "alt.atheism",
    "sci.space",
    "rec.sport.hockey",
    "talk.politics.guns",
]
TFIDF_MAX_FEATURES = 10_000
N_RETRIEVAL_QUERIES = 100
RETRIEVAL_K = 5
MIN_DOC_LENGTH = 20
RESULTS_LOG = os.path.join(os.path.dirname(__file__), "../../results/log.json")
# ──────────────────────────────────────────────────────────────────────────────


def filter_empty(docs, labels, min_len=MIN_DOC_LENGTH):
    pairs = [(d, l) for d, l in zip(docs, labels) if len(d.strip()) >= min_len]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def append_result(entry: dict) -> None:
    log_path = os.path.abspath(RESULTS_LOG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nResults saved → {log_path}")


def main():
    print("Loading 20 Newsgroups dataset...")
    train_docs, train_labels = load_20newsgroups(subset="train", categories=CATEGORIES)
    test_docs, test_labels = load_20newsgroups(subset="test", categories=CATEGORIES)

    empty_train = sum(1 for d in train_docs if len(d.strip()) < MIN_DOC_LENGTH)
    print(f"  Train: {len(train_docs)} docs ({empty_train} near-empty filtered)")
    train_docs, train_labels = filter_empty(train_docs, train_labels)
    test_docs, test_labels = filter_empty(test_docs, test_labels)
    print(f"  After filter — Train: {len(train_docs)} | Test: {len(test_docs)}")

    print(f"\nFitting TfidfEngine (max_features={TFIDF_MAX_FEATURES})...")
    engine = TfidfEngine(max_features=TFIDF_MAX_FEATURES, min_df=2)
    engine.fit(train_docs)

    print("Running SortingTask...")
    sorting_metrics = SortingTask(n_clusters=len(CATEGORIES)).run(
        engine, (train_docs, train_labels)
    )

    label_to_indices: dict = defaultdict(set)
    for idx, label in enumerate(train_labels):
        label_to_indices[label].add(idx)

    queries = [
        (test_docs[i], label_to_indices[test_labels[i]])
        for i in range(min(N_RETRIEVAL_QUERIES, len(test_docs)))
    ]

    print(f"Running RetrievalTask (k={RETRIEVAL_K})...")
    retrieval_metrics = RetrievalTask(k=RETRIEVAL_K).run(engine, (train_docs, queries))

    print("\n" + "=" * 50)
    print("BASELINE RESULTS — TF-IDF on 20 Newsgroups")
    print("=" * 50)
    print(f"  SortingTask   → ARI:            {sorting_metrics['ari']:.4f}")
    print(f"  RetrievalTask → Precision@{RETRIEVAL_K}:   {retrieval_metrics['precision_at_k']:.4f}")
    print(f"  RetrievalTask → MAP:            {retrieval_metrics['map']:.4f}")
    print(f"  RetrievalTask → MRR:            {retrieval_metrics['mrr']:.4f}")
    print("=" * 50)

    append_result({
        "date": str(date.today()),
        "engine": "TfidfEngine",
        "modality": "text",
        "dataset": "20newsgroups",
        "config": {"max_features": TFIDF_MAX_FEATURES},
        "metrics": {
            "sorting_ari": sorting_metrics["ari"],
            f"retrieval_precision_at_{RETRIEVAL_K}": retrieval_metrics["precision_at_k"],
            "retrieval_map": retrieval_metrics["map"],
            "retrieval_mrr": retrieval_metrics["mrr"],
        },
    })


if __name__ == "__main__":
    main()
