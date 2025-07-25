
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

class Visualizer:
    """
    Class responsible for visualizing clustering results.
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir (str): Directory where the visualizations will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_clusters(self, scaled_data: np.ndarray, labels: pd.Series, base_filename: str = 'cluster_plot') -> None:
        """
        Visualizes clustering results in a 2D space using PCA.

        Args:
            scaled_data (np.ndarray): Scaled data used for clustering.
            labels (pd.Series): Cluster labels for each data point.
            base_filename (str, optional): Base name for the output plot. Defaults to 'cluster_plot'.
        """
        print("Visualizing clustering results with PCA...")
        pca = PCA(n_components=2, random_state=42)
        data_pca = pca.fit_transform(scaled_data)

        plt.figure(figsize=(12, 10))
        
        # Draw noise points separately (if any)
        unique_labels = np.unique(labels)
        is_noise = -1 in unique_labels
        
        # Create a color map for clusters
        colors = plt.cm.get_cmap('viridis', len(unique_labels) - (1 if is_noise else 0))

        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points
                plt.scatter(data_pca[labels == label, 0], data_pca[labels == label, 1], 
                            s=10, color='gray', alpha=0.5, label='Noise')
            else:
                # Regular clusters
                plt.scatter(data_pca[labels == label, 0], data_pca[labels == label, 1], 
                            s=20, alpha=0.8, color=colors(i), label=f'Cluster {label}')

        plt.title('Molecule Clusters (PCA Visualization)', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"{base_filename}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualization saved to: {plot_path}")
