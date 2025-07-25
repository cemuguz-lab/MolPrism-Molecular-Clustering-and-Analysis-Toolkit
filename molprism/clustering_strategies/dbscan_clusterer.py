import pandas as pd
from sklearn.cluster import DBSCAN
from .base_clusterer import BaseClusterer

class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering strategy."""

    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame):
        super().__init__(original_data, scaled_data)
        self.model_ = None # We will store the DBSCAN model here

    def cluster(self, eps: float = 0.5, min_samples: int = 5, **kwargs) -> None:
        """
        Clusters the data using DBSCAN.

        Args:
            eps (float, optional): Maximum distance between two points to be considered neighbors. Defaults to 0.5.
            min_samples (int, optional): Number of samples in a neighborhood for a point to be considered a core point. Defaults to 5.
        """
        print(f"Creating DBSCAN model (eps={eps}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = dbscan.fit_predict(self.scaled_data)
        self.model_ = dbscan # Save the model
        print("Clustering completed.")