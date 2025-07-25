
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class DataPreprocessor:
    """
    Class responsible for loading, validating, cleaning, and preprocessing data for clustering.
    """

    def __init__(self, file_path: str, feature_columns: List[str]):
        """
        Args:
            file_path (str): Path to the input CSV file.
            feature_columns (List[str]): List of columns (features) to be used for clustering.
        """
        self.file_path = file_path
        self.feature_columns = feature_columns
        self.original_data = None
        self.scaled_data = None
        self.id_column = 'IDNUMBER'  # Default ID column, can be changed if necessary
        self.smiles_column = 'Canonical_SMILES'  # Default SMILES column

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the data, validates selected features, cleans and scales them.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Original data and scaled feature data.
        """
        print(f"Loading data from '{self.file_path}'...")
        try:
            df = pd.read_csv(self.file_path, low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found: {self.file_path}")

        # Check for the presence of all required feature columns
        required_cols = self.feature_columns + [self.id_column, self.smiles_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in the dataset: {missing_cols}")

        # Select only relevant columns
        self.original_data = df[required_cols].copy()

        print(f"Selected features: {self.feature_columns}")
        features_df = self.original_data[self.feature_columns].copy()

        # Data validation and cleaning
        print("Validating and cleaning data types...")
        initial_rows = len(features_df)
        for col in self.feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        features_df.dropna(inplace=True)
        cleaned_rows = len(features_df)
        
        if initial_rows > cleaned_rows:
            print(f"Warning: {initial_rows - cleaned_rows} rows with non-numeric or missing values were removed.")

        # Filter original data based on cleaned feature rows
        self.original_data = self.original_data.loc[features_df.index]

        if cleaned_rows == 0:
            raise ValueError("No data left for clustering after cleaning. Please check your feature selection.")

        # Feature scaling
        print("Standardizing features (StandardScaler)...")
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(features_df)

        print("Data preprocessing completed.")
        return self.original_data, self.scaled_data
