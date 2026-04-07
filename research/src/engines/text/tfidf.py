import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.engines.base import BaseEngine

class TfidfEngine(BaseEngine):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.is_fitted = False

    def fit(self, corpus):
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def embed_text(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before embedding")
        # transform returns a sparse matrix; convert to dense array
        return self.vectorizer.transform([text]).toarray()[0]
