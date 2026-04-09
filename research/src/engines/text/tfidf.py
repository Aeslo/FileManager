import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.engines.base import BaseEngine

class TfidfEngine(BaseEngine):
    def __init__(self, max_features: int = 10_000, min_df: int = 1, stop_words: str | None = 'english'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            sublinear_tf=True,
            min_df=min_df,
        )
        self.is_fitted = False

    def fit(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def embed_text(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before embedding")
        return self.vectorizer.transform([text]).toarray()[0] # type: ignore

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before embedding")
        return self.vectorizer.transform(texts).toarray() # type: ignore
