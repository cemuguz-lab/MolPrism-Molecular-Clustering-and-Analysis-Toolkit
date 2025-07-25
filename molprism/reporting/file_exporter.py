
import pandas as pd
import os

class FileExporter:
    """
    Class responsible for writing analysis results to files.
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir (str): Directory where the output files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_results(self, data_with_labels: pd.DataFrame, representatives: pd.DataFrame = None, base_filename: str = 'clustering_results') -> None:
        """
        Writes the clustered data and representative molecules to CSV files.

        Args:
            data_with_labels (pd.DataFrame): DataFrame containing the original data with an added 'Cluster' column.
            representatives (pd.DataFrame, optional): Selected representative molecules. Defaults to None.
            base_filename (str, optional): Base name for the output files. Defaults to 'clustering_results'.
        """
        # Save all data
        all_data_path = os.path.join(self.output_dir, f"{base_filename}_all.csv")
        data_with_labels.to_csv(all_data_path, index=False)
        print(f"All clustered data saved to: {all_data_path}")

        # Save representatives
        if representatives is not None and not representatives.empty:
            representatives_path = os.path.join(self.output_dir, f"{base_filename}_representatives.csv")
            representatives.to_csv(representatives_path, index=False)
            print(f"Representative molecules saved to: {representatives_path}")
