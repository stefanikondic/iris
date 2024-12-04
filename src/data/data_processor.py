import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, file_path, target):
        self.file_path = file_path
        self.target = target
        self.scaler = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        X = data.drop(columns=self.target)
        y = data[f"{self.target}"]
        return X, y

    def preprocess(self, scale=False, stratify_y=True):
        X, y = self.load_data()

        stratify = y if stratify_y else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
