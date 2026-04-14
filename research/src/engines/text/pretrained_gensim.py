import numpy as np
import gensim.downloader as api
from src.engines.base import BaseEngine
from src.engines.text.utils import tokenize

class GensimPretrainedEngine(BaseEngine):
    """Uses Gensim to load pre-trained GloVe/Word2Vec/FastText models."""
    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        print(f"Loading pre-trained model: {model_name} (this may take a while)...")
        # api.load() downloads and caches the model automatically
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size #type: ignore
        self.model_name = model_name

    def fit(self, corpus: list[str]) -> None:
        """Pre-trained models don't need fitting."""
        print(f"GensimPretrainedEngine ({self.model_name}) is ready (no fitting required).")

    def _get_vector(self, word: str) -> np.ndarray:
        if word in self.model:
            return self.model[word] # type: ignore
        return np.zeros(self.vector_size)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.vector_size)
        # Average the word vectors
        vecs = [self._get_vector(t) for t in tokens]
        return np.mean(vecs, axis=0)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts])
