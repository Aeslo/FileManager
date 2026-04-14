"""
Image baseline evaluation script.

Runs image engines against SortingTask and ImageRetrievalTask on STL-10,
then appends dated entries to results/log.json.

Usage:
    python -m src.evaluation.run_image_baseline
"""

import json
import os
from collections import defaultdict
from datetime import date

from src.data_utils.image_loader import load_stl10
from src.engines.image.color_histogram import ColorHistogramEngine
from src.engines.image.clip import CLIPEngine
from src.tasks.sorting import SortingTask
from src.tasks.image_retrieval import ImageRetrievalTask

# ── Config ────────────────────────────────────────────────────────────────────
N_RETRIEVAL_QUERIES = 50
RETRIEVAL_K = 5
# Caps images per class — set to None to use the full dataset
MAX_PER_CLASS_TRAIN = 100
MAX_PER_CLASS_TEST = 50
RESULTS_LOG = os.path.join(os.path.dirname(__file__), "../../results/log.json")
# ──────────────────────────────────────────────────────────────────────────────


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
    print(f"Results saved → {log_path}")


def main():
    print("Loading STL-10 dataset...")
    train_paths, train_labels = load_stl10(subset="train", max_per_class=MAX_PER_CLASS_TRAIN)
    test_paths, test_labels = load_stl10(subset="test", max_per_class=MAX_PER_CLASS_TEST)

    engines = {
        #"ColorHistogram": ColorHistogramEngine(bins=32),
        "CLIP-ViT-B32": CLIPEngine("openai/clip-vit-base-patch32"),
    }

    for name, engine in engines.items():
        print("\n" + "=" * 50)
        print(f"RUNNING BENCHMARK: {name}")
        print("=" * 50)

        engine.fit(train_paths)

        print("Running SortingTask...")
        sorting_metrics = SortingTask(n_clusters=10).run(engine, (train_paths, train_labels))

        label_to_indices: dict = defaultdict(set)
        for idx, label in enumerate(train_labels):
            label_to_indices[label].add(idx)

        queries = [
            (test_paths[i], label_to_indices[test_labels[i]])
            for i in range(min(N_RETRIEVAL_QUERIES, len(test_paths)))
        ]

        print(f"Running ImageRetrievalTask (k={RETRIEVAL_K})...")
        retrieval_metrics = ImageRetrievalTask(k=RETRIEVAL_K).run(engine, (train_paths, queries))

        print(f"\nRESULTS for {name}:")
        print(f"  SortingTask   → ARI:            {sorting_metrics['ari']:.4f}")
        print(f"  RetrievalTask → Precision@{RETRIEVAL_K}:   {retrieval_metrics['precision_at_k']:.4f}")
        print(f"  RetrievalTask → MAP:            {retrieval_metrics['map']:.4f}")
        print(f"  RetrievalTask → MRR:            {retrieval_metrics['mrr']:.4f}")

        append_result({
            "date": str(date.today()),
            "engine": name,
            "modality": "image",
            "dataset": "stl10",
            "config": {
                "vector_size": getattr(engine, "vector_size", None),
            },
            "metrics": {
                "sorting_ari": sorting_metrics["ari"],
                f"retrieval_precision_at_{RETRIEVAL_K}": retrieval_metrics["precision_at_k"],
                "retrieval_map": retrieval_metrics["map"],
                "retrieval_mrr": retrieval_metrics["mrr"],
            },
        })


if __name__ == "__main__":
    main()
