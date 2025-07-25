import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from .base_clusterer import BaseClusterer

class AgglomerativeClusterer(BaseClusterer):
    """Hierarchical Agglomerative clustering strategy."""

    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame):
        super().__init__(original_data, scaled_data)
        self.model_ = None # We'll store the AgglomerativeClustering model here

    def cluster(self, n_clusters: int = 2, linkage: str = 'ward', **kwargs) -> None:
        """
        Clusters the data using Agglomerative Clustering.

        Args:
            n_clusters (int, optional): Number of clusters to create. Defaults to 2.
            linkage (str, optional): Which linkage criterion to use ('ward', 'complete', 'average', 'single'). Defaults to 'ward'.
        """
        print(f"Creating Agglomerative Clustering model (n_clusters={n_clusters}, linkage='{linkage}')...")
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = agg.fit_predict(self.scaled_data)
        self.model_ = agg # Save the model
        print("Clustering completed.")