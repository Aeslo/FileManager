import numpy as np
import pytest
from src.engines.text.tfidf import TfidfEngine

def test_tfidf_engine_embed():
    engine = TfidfEngine()
    corpus = ["hello world", "this is a test"]
    engine.fit(corpus)
    vec = engine.embed_text("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] > 0
    # Check if a non-corpus word still gives a vector (though it might be all zeros)
    vec_unk = engine.embed_text("unknownword")
    assert isinstance(vec_unk, np.ndarray)

def test_tfidf_engine_not_fitted():
    engine = TfidfEngine()
    with pytest.raises(ValueError, match="Engine must be fitted before embedding"):
        engine.embed_text("fail")

def test_tfidf_engine_embed_batch():
    corpus = ["hello world", "this is a test", "foo bar baz"]
    engine = TfidfEngine()
    engine.fit(corpus)
    vecs = engine.embed_batch(corpus)
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape[0] == len(corpus)


def test_tfidf_engine_max_features():
    corpus = ["word" + str(i) for i in range(200)]
    engine = TfidfEngine(max_features=50)
    engine.fit(corpus)
    vec = engine.embed_text(corpus[0])
    assert vec.shape[0] == 50


def test_tfidf_engine_batch_not_fitted():
    engine = TfidfEngine()
    with pytest.raises(ValueError, match="Engine must be fitted before embedding"):
        engine.embed_batch(["fail"])
