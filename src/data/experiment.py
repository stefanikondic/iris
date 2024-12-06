import logging
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import yaml
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log"),
    ],
)

SUPPORTED_MODELS = {
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
}


class Experiment:
    def __init__(self, model_config_path, output_dir):
        self.model_config_path = model_config_path
        self.output_dir = output_dir
        self.model = None
        self.best_params = None

        os.makedirs(os.path.join(self.output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)

        logging.info(
            "Experiment initialized with config: %s and output_dir: %s",
            self.model_config_path,
            self.output_dir,
        )

        print(SUPPORTED_MODELS.keys())

    def load_model_config(self):
        try:
            with open(self.model_config_path, "r") as file:
                config = yaml.safe_load(file)

            logging.info("Model configuration loaded successfully.")

            return config

        except FileNotFoundError:
            logging.error("Config file not found: %s", self.model_config_path)
            raise

        except yaml.YAMLError as e:
            logging.error("Error parsing YAML config: %s", e)
            raise

    def initialize_model(self, grid_search=False):
        config = self.load_model_config()
        model_name = config.get("model", {}).get("name")
        params = config.get("model", {}).get("parameters", {})

        if not model_name:
            logging.error("Model name is missing in the configuration file.")
            raise ValueError("Model name is missing in the configuration file.")

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models are: {list(SUPPORTED_MODELS.keys())}"
            )

        self.model = SUPPORTED_MODELS[model_name]

        if grid_search:
            logging.info("Performing Grid Search for model: %s", model_name)

            grid_search = GridSearchCV(
                self.model, params, cv=5, scoring="accuracy", verbose=4
            )
            self.model = grid_search
        else:
            try:
                self.model.set_params(**params)
                logging.info("Parameters set for %s: %s", model_name, params)
            except ValueError as e:
                logging.error("Error setting model parameters: %s", e)
                raise

    def train(self, X_train, y_train, grid_search=False):
        if not self.model:
            logging.debug("Initializing model with grid_search=%s", grid_search)
            self.initialize_model(grid_search=grid_search)

        if isinstance(self.model, GridSearchCV):
            logging.info("Training model with Grid Search...")
            self.model.fit(X_train, y_train)
            self.best_params = self.model.best_params_
            logging.info(
                "Best parameters for %s: %s",
                self.model.estimator.__class__.__name__,
                self.best_params,
            )
            self.model = self.model.best_estimator_
        else:
            logging.info("Training model without Grid Search...")
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        logging.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        output_path = os.path.join(
            self.output_dir,
            "metrics",
            os.path.basename(self.model_config_path).replace(".yaml", "_metrics.json"),
        )

        try:
            with open(output_path, "w") as file:
                json.dump(metrics, file, indent=4)
            logging.info("Metrics saved to %s", output_path)
        except IOError as e:
            logging.error("Error saving metrics: %s", e)
            raise

        return metrics

    def save_model(self, model_name):
        model_path = os.path.join(self.output_dir, "models", f"{model_name}.pkl")
        try:
            joblib.dump(self.model, model_path)
            logging.info("Model saved to %s", model_path)
        except IOError as e:
            logging.error("Error saving model: %s", e)
            raise
