import numpy as np
import pytest
from src.engines.text.fasttext import FastTextEngine, _ngrams


CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are great pets",
    "the sky is blue and the grass is green",
    "space exploration is exciting and important",
    "the astronauts flew to the moon last year",
]


# ── Unit tests for n-gram helper ──────────────────────────────────────────────

def test_ngrams_produces_correct_grams():
    grams = _ngrams("cat", min_n=3, max_n=4)
    # padded = "<cat>", 3-grams: "<ca", "cat", "at>", 4-grams: "<cat", "cat>"
    assert "<ca" in grams
    assert "cat" in grams
    assert "at>" in grams
    assert "<cat" in grams


def test_ngrams_min_equals_max():
    grams = _ngrams("hi", min_n=3, max_n=3)
    # "<hi>" length=4, only one 3-gram: "<hi"? let's check
    padded = "<hi>"
    expected = [padded[i:i+3] for i in range(len(padded) - 3 + 1)]
    assert grams == expected


# ── Engine tests ───────────────────────────────────────────────────────────────

def test_embed_returns_ndarray():
    engine = FastTextEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    vec = engine.embed_text("cat sat on mat")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_embed_vector_size():
    engine = FastTextEngine(vector_size=20, min_count=1)
    engine.fit(CORPUS)
    vec = engine.embed_text("cat")
    assert vec.shape[0] == 20


def test_embed_batch_shape():
    engine = FastTextEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    texts = ["cat sat", "dog ran", "sky is blue"]
    vecs = engine.embed_batch(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 10)


def test_not_fitted_raises():
    engine = FastTextEngine()
    with pytest.raises(ValueError, match="Engine must be fitted before embedding"):
        engine.embed_text("hello")


def test_oov_handled_via_subwords():
    """FastText key property: OOV words get a non-zero vector via their n-grams."""
    engine = FastTextEngine(vector_size=10, min_count=1)
    engine.fit(CORPUS)
    # "catz" was never in the corpus but shares n-grams with "cat"
    vec = engine.embed_text("catz")
    assert isinstance(vec, np.ndarray)
    # Should NOT be all zeros — n-grams of "catz" overlap with known vocab
    assert not np.allclose(vec, 0), "FastText should give a non-zero vector for OOV words via subwords"


def test_oov_similar_to_known_word():
    """Misspelling should be close to the correct word because they share n-grams."""
    engine = FastTextEngine(vector_size=30, min_count=1)
    engine.fit(CORPUS)

    v_cat = engine.embed_text("cat")
    v_cats_typo = engine.embed_text("caat")   # typo — shares n-grams with "cat"
    v_space = engine.embed_text("space")

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    sim_typo = cosine(v_cat, v_cats_typo)
    sim_unrelated = cosine(v_cat, v_space)
    assert sim_typo >= sim_unrelated, (
        f"Expected typo similarity ({sim_typo:.3f}) >= "
        f"unrelated similarity ({sim_unrelated:.3f})"
    )
