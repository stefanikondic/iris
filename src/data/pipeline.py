import json
import os

from colorama import Fore, Style
import pandas as pd

from src.data.data_processor import DataProcessor
from src.data.experiment import Experiment


class Pipeline:
    """
    A class to orchestrate the machine learning workflow, including data preprocessing,
    model training, evaluation, and reporting.

    Attributes:
        data_processor (DataProcessor): Instance of DataProcessor to handle data loading and preprocessing.
        experiments_dir (str): Path to the directory containing experiment configuration files.
        output_dir (str): Path to the directory for saving outputs (e.g., models, metrics, reports).
        report (list): List to store results and metrics for each experiment.
    """

    def __init__(self, data_path, target, experiments_dir, output_dir):
        """
        Initializes the Pipeline with paths for data, experiments, and output directories.

        Args:
            data_path (str): Path to the dataset file.
            target (str): The name of the target column in the dataset.
            experiments_dir (str): Path to the directory containing YAML experiment configurations.
            output_dir (str): Path to the directory for saving outputs (e.g., metrics, models).
        """

        self.data_processor = DataProcessor(data_path, target)
        self.experiments_dir = experiments_dir
        self.output_dir = output_dir
        self.report = []

    def run(self, scale=False, grid_search=False):
        """
        Executes the entire machine learning pipeline, including preprocessing,
        training, evaluation, and report generation.

        Args:
            scale (bool): Whether to scale features using StandardScaler during preprocessing. Default is False.
            grid_search (bool): Whether to perform hyperparameter tuning using GridSearchCV. Default is False.

        Returns:
            None

        Raises:
            FileNotFoundError: If the experiments directory is missing or empty.
        """

        X_train, X_test, y_train, y_test = self.data_processor.preprocess(scale=scale)

        for config_file in os.listdir(self.experiments_dir):
            if config_file.endswith(".yaml"):
                config_path = os.path.join(self.experiments_dir, config_file)
                experiment = Experiment(config_path, self.output_dir)
                print(f"Type of exp: {type(experiment)}")

                experiment.train(X_train, y_train, grid_search=grid_search)
                metrics = experiment.evaluate(X_test, y_test)
                experiment.save_model(config_file.split(".")[0])
                self.report.append(
                    {
                        "experiment": config_file.split(".")[0],
                        "metrics": metrics,
                        "best_params": experiment.best_params if grid_search else None,
                    }
                )

        self.save_report()
        self.generate_csv_report()

    def save_report(self):
        report_path = os.path.join(self.output_dir, "summary_report.json")
        with open(report_path, "w") as file:
            json.dump(self.report, file, indent=4)
            print(f"Consolidated report saved to {report_path}")

    def generate_csv_report(self):
        report_csv_path = os.path.join(self.output_dir, "summary_report.csv")
        flat_report = []

        for entry in self.report:
            row = {
                "experiment": entry["experiment"],
                "accuracy": entry["metrics"]["accuracy"],
                "best_params": entry["best_params"],
            }
            flat_report.append(row)

        df = pd.DataFrame(flat_report)
        df.to_csv(report_csv_path, index=False)
        print(f"CSV report saved to {report_csv_path}")
