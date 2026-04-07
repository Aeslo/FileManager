from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.tasks.base import BaseTask
import numpy as np

class SortingTask(BaseTask):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def run(self, engine, dataset) -> dict:
        data, true_labels = dataset
        # Embed each item in the dataset
        embeddings = np.array([engine.embed_text(text) for text in data])
        
        # Simple K-Means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(embeddings)
        
        # Calculate Adjusted Rand Index (ARI)
        ari = adjusted_rand_score(true_labels, pred_labels)
        return {"ari": float(ari)}
