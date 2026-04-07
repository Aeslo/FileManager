from sklearn.datasets import fetch_20newsgroups

def load_20newsgroups(subset='train', categories=None):
    newsgroups = fetch_20newsgroups(subset=subset, categories=categories, remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target
