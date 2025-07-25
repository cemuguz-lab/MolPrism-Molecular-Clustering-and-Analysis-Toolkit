
from abc import ABC, abstractmethod
import pandas as pd

class BaseClusterer(ABC):
    """
    Abstract base class (interface) for all clustering strategies.
    Every clustering algorithm must inherit from this class and
    implement the 'cluster' method.
    """

    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame):
        self.original_data = original_data
        self.scaled_data = scaled_data
        self.labels_ = None

    @abstractmethod
    def cluster(self, **kwargs) -> None:
        """
        Clusters the data and stores the labels in 'self.labels_'.

        Args:
            **kwargs: Additional algorithm-specific parameters (e.g., n_clusters).
        """
        pass
