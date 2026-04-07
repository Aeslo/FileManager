from src.data_utils.loader import load_20newsgroups

def test_load_20newsgroups():
    # Only load a tiny subset for testing purposes
    data, labels = load_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'])
    assert len(data) > 0
    assert len(data) == len(labels)
    assert isinstance(data[0], str)
