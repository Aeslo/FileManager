import numpy as np
import pytest
from src.engines.text.word2vec import Word2VecEngine


CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are great pets",
    "the sky is blue and the grass is green",
    "space exploration is exciting and important",
    "the astronauts flew to the moon last year",
]


def test_embed_returns_ndarray():
    engine = Word2VecEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    vec = engine.embed_text("cat sat on mat")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_embed_vector_size():
    engine = Word2VecEngine(vector_size=20, min_count=1)
    engine.fit(CORPUS)
    vec = engine.embed_text("cat")
    assert vec.shape[0] == 20


def test_embed_batch_shape():
    engine = Word2VecEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    texts = ["cat sat", "dog ran", "sky is blue"]
    vecs = engine.embed_batch(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 10)


def test_not_fitted_raises():
    engine = Word2VecEngine()
    with pytest.raises(ValueError, match="Engine must be fitted before embedding"):
        engine.embed_text("hello")


def test_oov_returns_zero_vector():
    """A document with only unknown words should return a zero vector, not crash."""
    engine = Word2VecEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    vec = engine.embed_text("zzzzunknownwordzzz")
    assert isinstance(vec, np.ndarray)
    assert np.allclose(vec, 0)


def test_similar_docs_closer_than_dissimilar():
    """Documents about the same topic should be closer than unrelated ones."""
    engine = Word2VecEngine(vector_size=30, window=3, min_count=1)
    engine.fit(CORPUS)

    v_cat = engine.embed_text("cat sat on mat")
    v_dog = engine.embed_text("dog ran in park")
    v_space = engine.embed_text("astronauts flew to moon")

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # cat vs dog (both animals/pets context) should beat cat vs space
    sim_cat_dog = cosine(v_cat, v_dog)
    sim_cat_space = cosine(v_cat, v_space)
    assert sim_cat_dog >= sim_cat_space, (
        f"Expected cat-dog similarity ({sim_cat_dog:.3f}) >= "
        f"cat-space similarity ({sim_cat_space:.3f})"
    )
