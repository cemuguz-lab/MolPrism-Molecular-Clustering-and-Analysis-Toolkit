
import pandas as pd
from typing import Type

from molprism.clustering_strategies.base_clusterer import BaseClusterer
from molprism.clustering_strategies.kmeans_clusterer import KMeansClusterer
from molprism.clustering_strategies.dbscan_clusterer import DBSCANClusterer
from molprism.clustering_strategies.agglomerative_clusterer import AgglomerativeClusterer

# Collect available strategies in a dictionary
AVAILABLE_STRATEGIES = {
    "kmeans": KMeansClusterer,
    "dbscan": DBSCANClusterer,
    "agglomerative": AgglomerativeClusterer,
}

class ClusteringManager:
    """
    Manager class responsible for selecting and executing the appropriate clustering strategy.
    """

    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame):
        self.original_data = original_data
        self.scaled_data = scaled_data
        self.clusterer: BaseClusterer = None

    def set_strategy(self, strategy_name: str) -> None:
        """
        Sets the clustering strategy to be used.

        Args:
            strategy_name (str): Name of the selected strategy (e.g., 'kmeans').

        Raises:
            ValueError: If the selected strategy is not available.
        """
        strategy_name = strategy_name.lower()
        if strategy_name not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy_name}. Available strategies: {list(AVAILABLE_STRATEGIES.keys())}")
        
        print(f"Clustering strategy '{strategy_name}' has been selected.")
        self.clusterer = AVAILABLE_STRATEGIES[strategy_name](self.original_data, self.scaled_data)

    def run_clustering(self, **kwargs) -> pd.Series:
        """
        Runs the clustering process using the selected strategy.

        Args:
            **kwargs: Parameters specific to the clustering algorithm.

        Returns:
            pd.Series: Cluster labels.
        """
        if not self.clusterer:
            raise RuntimeError("You must first set a strategy. Use the 'set_strategy' method.")
        
        self.clusterer.cluster(**kwargs)
        return self.clusterer.labels_
