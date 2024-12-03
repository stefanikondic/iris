from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json


def evaluate_model(model, X_test, y_test, output_path):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {"accuracy": accuracy, "classification_report": report}

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
