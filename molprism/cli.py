
import argparse
import time
import yaml
import os

from molprism.core.data_preprocessor import DataPreprocessor
from molprism.core.clustering_manager import ClusteringManager, AVAILABLE_STRATEGIES
from molprism.analysis.representative_selector import RepresentativeSelector
from molprism.reporting.file_exporter import FileExporter
from molprism.reporting.visualizer import Visualizer

def load_config(config_path):
    """Loads settings from a YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main CLI function for MolPrism."""
    parser = argparse.ArgumentParser(
        description="MolPrism: A Molecular Clustering and Analysis Toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Primary config argument
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file. Command-line arguments will override settings from this file.")

    # Input/Output Arguments
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files.")
    parser.add_argument("--features", nargs='+', help="List of features (columns) to use for clustering.")

    # Algorithm Arguments
    parser.add_argument("--algorithm", type=str, choices=AVAILABLE_STRATEGIES.keys(), help="Clustering algorithm to use.")
    parser.add_argument("--n_representatives", type=int, help="Number of representative molecules to select from each cluster.")

    # KMeans/Agglomerative specific
    parser.add_argument("--n_clusters", type=int, help="Number of clusters for KMeans or Agglomerative Clustering.")
    parser.add_argument("--find_best_k", action="store_true", help="Automatically find the best k for KMeans.")

    # DBSCAN specific
    parser.add_argument("--eps", type=float, help="DBSCAN for eps value.")
    parser.add_argument("--min_samples", type=int, help="DBSCAN for min_samples value.")

    # Agglomerative specific
    parser.add_argument("--linkage", type=str, help="Agglomerative Clustering for linkage method.")

    # Performance
    parser.add_argument("--n_jobs", type=int, help="Number of CPU cores to use for parallel tasks (-1 for all).")

    args = parser.parse_args()

    # Load config from file if specified
    config = {}
    if args.config:
        config = load_config(args.config)

    # --- Merge settings: command-line > config file > defaults ---
    # Helper to resolve priority
    def get_setting(arg_name, default=None):
        cmd_arg = getattr(args, arg_name)
        if cmd_arg is not None:
            return cmd_arg
        return config.get(arg_name, default)

    input_file = get_setting('input_file')
    output_dir = get_setting('output_dir')
    features = get_setting('features')
    algorithm = get_setting('algorithm', 'kmeans')
    n_representatives = get_setting('n_representatives', 10)
    n_jobs = get_setting('n_jobs', -1)

    # Check for required settings
    if not all([input_file, output_dir, features]):
        raise ValueError("Missing required settings: input_file, output_dir, and features must be provided either via command line or config file.")

    start_time = time.time()
    print("MolPrism analysis process started...")

    try:
        # 1. Data Preprocessing
        preprocessor = DataPreprocessor(file_path=input_file, feature_columns=features)
        original_data, scaled_data = preprocessor.load_and_prepare_data()

        # 2. Clustering
        manager = ClusteringManager(original_data, scaled_data)
        manager.set_strategy(algorithm)

        # Get algorithm-specific params from the config file or use defaults
        algo_params = config.get(f"{algorithm}_params", {})
        
        # Override with specific command-line args if provided
        if algorithm == 'kmeans':
            if args.n_clusters: algo_params['n_clusters'] = args.n_clusters
            if args.find_best_k: algo_params['find_best_k'] = True
            # Ensure n_clusters is set if not finding best k
            if not algo_params.get('find_best_k') and 'n_clusters' not in algo_params:
                raise ValueError("For KMeans, either --n_clusters or --find_best_k must be provided.")

        elif algorithm == 'dbscan':
            if args.eps: algo_params['eps'] = args.eps
            if args.min_samples: algo_params['min_samples'] = args.min_samples

        elif algorithm == 'agglomerative':
            if args.n_clusters: algo_params['n_clusters'] = args.n_clusters
            if args.linkage: algo_params['linkage'] = args.linkage
            if 'n_clusters' not in algo_params:
                raise ValueError("For Agglomerative Clustering, --n_clusters must be provided.")
        
        algo_params['n_jobs'] = n_jobs # Add n_jobs to every algorithm's params

        labels = manager.run_clustering(**algo_params)
        original_data['Cluster'] = labels

        # 3. Analysis (Representative Selection)
        selector = RepresentativeSelector(original_data, scaled_data, labels, manager.clusterer.model_)
        representatives = selector.select_representatives(n_representatives)

        # 4. Reporting
        base_filename = f"{algorithm}_results"
        exporter = FileExporter(output_dir)
        exporter.export_results(original_data, representatives, base_filename)

        visualizer = Visualizer(output_dir)
        visualizer.plot_clusters(scaled_data, labels, base_filename)

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"MolPrism analysis process finished in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
