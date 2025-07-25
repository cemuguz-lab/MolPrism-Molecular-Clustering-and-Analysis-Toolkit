import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

class RepresentativeSelector:
    """
    Selects representative points from each cluster based on clustering results.
    """


    def __init__(self, original_data: pd.DataFrame, scaled_data: np.ndarray, labels: pd.Series, clusterer_model=None):
        """
        Args:
            original_data (pd.DataFrame): Original, unscaled data.
            scaled_data (np.ndarray): Scaled data used in clustering.
            labels (pd.Series): Cluster labels for each data point.
            clusterer_model: The clustering model used (e.g., KMeans model).
        """
        self.original_data = original_data
        self.scaled_data = scaled_data
        self.labels = labels
        self.clusterer_model = clusterer_model

    def select_representatives(self, n_representatives: int) -> pd.DataFrame:
        """
        Selects the specified number of representatives from each cluster.
        Primarily tries a center-based method (such as KMeans).
        If the model is not center-based, it switches to a general distance-based method.

        Args:
            n_representatives (int): Number of representatives to select from each cluster.

        Returns:
            pd.DataFrame: A DataFrame containing all selected representatives.
        """
        print(f"Selecting {n_representatives} representative molecules from each cluster...")
        
        # If the model has the 'cluster_centers_' attribute (like KMeans)
        if self.clusterer_model is not None and hasattr(self.clusterer_model, 'cluster_centers_'):
            return self._select_nearest_to_centers(n_representatives)
        else:
            # General approach for non-center-based algorithms (e.g., DBSCAN)
            print("Warning: A non-center-based algorithm is being used or the model does not provide center information. Representatives will be selected based on the medoid of each cluster.")
            return self._select_medoids(n_representatives)

    def _select_nearest_to_centers(self, n_representatives: int) -> pd.DataFrame:
        """
        Selects the N points closest to each cluster center.
        """
        centers = self.clusterer_model.cluster_centers_
        selected_indices = []

        for i in range(len(centers)):
            # Get the indices of all points in the cluster
            cluster_indices = np.where(self.labels == i)[0]
            if len(cluster_indices) == 0:
                continue

            # Get the scaled data for that cluster
            cluster_scaled_data = self.scaled_data[cluster_indices]
            
            # Compute distances of the cluster points to the cluster center
            distances = pairwise_distances(cluster_scaled_data, [centers[i]])
            
            # Sort by distance and get the indices of the N closest points
            # If the number of elements in the cluster is less than requested, select all
            num_to_select = min(n_representatives, len(cluster_indices))
            nearest_indices_in_cluster = np.argsort(distances.ravel())[:num_to_select]
            
            # Retrieve the actual indices in the original dataset
            selected_indices.extend(cluster_indices[nearest_indices_in_cluster])

        return self.original_data.iloc[selected_indices].copy()

    def _select_medoids(self, n_representatives: int) -> pd.DataFrame:
        """
        Selects the medoid of each cluster (the point with the smallest average distance to other points)
        and its nearest neighbors.
        """
        selected_indices = []
        unique_labels = np.unique(self.labels[self.labels != -1]) # Exclude noise points (-1)

        for label in unique_labels:
            cluster_indices = np.where(self.labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_scaled_data = self.scaled_data[cluster_indices]
            
            # Compute intra-cluster distances
            distances = pairwise_distances(cluster_scaled_data)
            
            # Find the medoid (the point with the smallest total distance)
            medoid_index_in_cluster = np.argmin(distances.sum(axis=0))
            
            # Sort by distance to the medoid and select the N closest points
            num_to_select = min(n_representatives, len(cluster_indices))
            nearest_indices_in_cluster = np.argsort(distances[medoid_index_in_cluster])[:num_to_select]

            selected_indices.extend(cluster_indices[nearest_indices_in_cluster])

        return self.original_data.iloc[selected_indices].copy()