from sklearn.model_selection import cross_val_score
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib


def train_model(X_train, y_train, config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]
    parameters = config["model"]["parameters"]

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**parameters)
    elif model_name == "SVC":
        model = SVC(**parameters)
    elif model_name == "KNeighborsClassifier":
        model = KNeighborsClassifier(**parameters)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores for {model_name}: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean()}")

    joblib.dump(model, f"outputs/models/{model_name}.pkl")
    return model
