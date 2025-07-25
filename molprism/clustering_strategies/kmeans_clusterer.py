
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from joblib import Parallel, delayed
import numpy as np

from .base_clusterer import BaseClusterer

def _calculate_k_metrics(data, k):
    """Runs KMeans for a given k value and computes metrics (for parallel execution)."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(data)
    inertia = kmeans.inertia_
    try:
        score = silhouette_score(data, kmeans.labels_)
    except ValueError:
        score = -1  # For invalid clustering cases
    return k, inertia, score

class KMeansClusterer(BaseClusterer):
    """KMeans clustering strategy."""

    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame):
        super().__init__(original_data, scaled_data)
        self.model_ = None

    def cluster(self, n_clusters: int = None, find_best_k: bool = False, min_k: int = 2, max_k: int = 20, n_jobs: int = -1) -> None:
        """
        Clusters the data using KMeans.

        Args:
            n_clusters (int, optional): Number of clusters to use. If not specified, 'find_best_k' must be True.
            find_best_k (bool, optional): Automatically finds the best k value. Defaults to False.
            min_k (int, optional): Minimum number of clusters to search for best k. Defaults to 2.
            max_k (int, optional): Maximum number of clusters to search for best k. Defaults to 20.
            n_jobs (int, optional): Number of threads to use for parallel processing. Defaults to -1 (all cores).
        """
        if n_clusters is None and not find_best_k:
            raise ValueError("If 'n_clusters' is not specified, 'find_best_k' must be True.")

        if find_best_k:
            print(f"Searching for the best number of clusters (k) in the range {min_k}-{max_k} in parallel...")
            results = Parallel(n_jobs=n_jobs)(
                delayed(_calculate_k_metrics)(self.scaled_data, k) for k in range(min_k, max_k + 1)
            )
            
            k_values, inertia, silhouette_scores = zip(*results)
            
            # Elbow method
            knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
            elbow_k = knee_locator.knee
            
            # Silhouette score
            best_silhouette_k = k_values[np.argmax(silhouette_scores)]
            
            print(f"Recommended k by Elbow method: {elbow_k}")
            print(f"Recommended k by Silhouette Score: {best_silhouette_k}")
            
            # The elbow method generally provides a more reliable upper bound.
            n_clusters = elbow_k if elbow_k else best_silhouette_k
            print(f"Selected best number of clusters: {n_clusters}")

        print(f"Creating KMeans model with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.labels_ = kmeans.fit_predict(self.scaled_data)
        self.model_ = kmeans
        print("Clustering completed.")

