import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path, scale=False, add_noise=False):
    data = pd.read_csv(file_path)
    X = data.drop(columns="species")
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:\n", y_train.value_counts())
    print("y_test distribution:\n", y_test.value_counts())

    if add_noise:
        # Add Gaussian noise to the dataset
        np.random.seed(42)
        X_train += np.random.normal(0, 0.1, X_train.shape)
        X_test += np.random.normal(0, 0.1, X_test.shape)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
