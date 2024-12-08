import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    A class to handle data loading, preprocessing, and splitting for machine learning tasks.

    Attributes:
        file_path (str): Path to the CSV file containing the dataset.
        target (str): The name of the target column in the dataset.
        scaler (StandardScaler or None): A scaler object used for feature scaling, if applicable.
    """

    def __init__(self, file_path, target):
        """
        Initializes the DataProcessor with the dataset path and target column.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
            target (str): The name of the target column in the dataset.
        """
        self.file_path = file_path
        self.target = target
        self.scaler = None

    def load_data(self):
        """
        Loads the dataset from the specified file path and separates features and target.

        Returns:
            tuple:
                - X (pd.DataFrame): The feature columns of the dataset.
                - y (pd.Series): The target column of the dataset.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.
            KeyError: If the target column is not found in the dataset.
        """

        data = pd.read_csv(self.file_path)
        X = data.drop(columns=self.target)
        y = data[f"{self.target}"]
        return X, y

    def preprocess(self, scale=False, stratify_y=True):
        """
        Preprocesses the dataset by splitting it into train and test sets, and optionally scales the features.

        Args:
            scale (bool): Whether to scale the features using StandardScaler. Default is False.
            stratify_y (bool): Whether to stratify the train-test split based on the target variable. Default is True.

        Returns:
            tuple:
                - X_train (array-like): Training feature set.
                - X_test (array-like): Testing feature set.
                - y_train (array-like): Training target labels.
                - y_test (array-like): Testing target labels.

        Raises:
            ValueError: If stratification is enabled and the target variable has too few samples for stratified splitting.
        """

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
