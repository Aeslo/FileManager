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
