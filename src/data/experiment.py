import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json


class Experiment:
    def __init__(self, model_config_path, output_dir):
        self.model_config_path = model_config_path
        self.output_dir = output_dir
        self.model = None

    def load_model_config(self):
        with open(self.model_config_path, "r") as file:
            config = yaml.safe_load(file)

        return config

    def initialize_model(self):
        config = self.load_model_config()
        model_name = config["model"]["name"]
        params = config["model"]["parameters"]

        if model_name == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(**params)
        elif model_name == "SVC":
            from sklearn.svm import SVC

            self.model = SVC(**params)
        elif model_name == "KNeighborsClassifier":
            from sklearn.neighbors import KNeighborsClassifier

            self.model = KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def train(self, X_train, y_train):
        if not self.model:
            self.initialize_model()
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        output_path = f"{self.output_dir}/metrics/{self.model_config_path.split('/')[-1].replace('.yaml', '_metrics.json')}"

        with open(output_path, "w") as file:
            json.dump(metrics, file, indent=4)

        print(f"Metrics saved to {output_path}")
        return metrics

    def save_model(self, model_name):
        model_path = f"{self.output_dir}/models/{model_name}.pkl"
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
