import os

from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.models.evaluate import evaluate_model
from src.models.train_model import train_model

DATA_PATH = "data/raw/iris.csv"
EXPERIMENTS_PATH = "experiments"
OUTPUT_METRICS_PATH = "outputs/metrics"


def run_all_experiments():
    list_of_experiments = os.listdir(EXPERIMENTS_PATH)
    experiments = [
        experiment_config
        for experiment_config in list_of_experiments
        if experiment_config.endswith(".yaml")
    ]

    for experiment in experiments:
        config_file_path = os.path.join(EXPERIMENTS_PATH, experiment)
        X_train, X_test, y_train, y_test = preprocess_data(
            DATA_PATH, scale=True, add_noise=True
        )

        model = train_model(X_train, y_train, config_file_path)

        output_path = os.path.join(
            OUTPUT_METRICS_PATH, f"{experiment.split('.')[0]}_metrics.json"
        )
        metrics = evaluate_model(model, X_test, y_test, output_path)

        print(
            f"Experiment {experiment.split('.')[0]} complete. Metrics saved to {output_path}"
        )


if __name__ == "__main__":
    load_data()
    run_all_experiments()
