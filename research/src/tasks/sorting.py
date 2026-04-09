from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from src.tasks.base import BaseTask
import numpy as np

class SortingTask(BaseTask):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def run(self, engine, dataset: tuple[list[str], list[int] | np.ndarray]) -> dict[str, float]:
        data, true_labels = dataset
        embeddings = engine.embed_batch(data)

        # L2-normalize so K-Means (Euclidean) approximates cosine distance.
        # Critical for sparse high-dim vectors like TF-IDF.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero for empty docs
        embeddings = embeddings / norms

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(embeddings)
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        return {"ari": float(ari)}
