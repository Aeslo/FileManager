import sys
import os

# Ensure the directory containing this file is in sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_utils.loader import load_20newsgroups
from src.engines.text.tfidf import TfidfEngine
import numpy as np

CATEGORIES = [
    "alt.atheism",
    "sci.space",
    "rec.sport.hockey",
    "talk.politics.guns",
]

def main():
    print("Loading data...")
    train_docs, train_labels = load_20newsgroups(subset="train", categories=CATEGORIES)
    print(f"Loaded {len(train_docs)} docs")
    
    empty_docs = [d for d in train_docs if not d.strip()]
    print(f"Empty docs: {len(empty_docs)} ({len(empty_docs)/len(train_docs)*100:.2f}%)")
    
    short_docs = [d for d in train_docs if len(d.strip()) < 20]
    print(f"Short docs (<20 chars): {len(short_docs)} ({len(short_docs)/len(train_docs)*100:.2f}%)")

    engine = TfidfEngine(max_features=1000)
    engine.fit(train_docs)
    
    embeddings = engine.embed_batch(train_docs[:100])
    
    zero_rows = np.all(embeddings == 0, axis=1)
    print(f"Zero rows in first 100 embeddings: {np.sum(zero_rows)}")
    
    non_zero_elements = np.count_nonzero(embeddings)
    print(f"Non-zero elements in first 100 embeddings: {non_zero_elements}")

if __name__ == "__main__":
    main()
