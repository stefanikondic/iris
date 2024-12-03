import pandas as pd
from sklearn.datasets import load_iris


def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["species"] = iris.target
    data.to_csv("data/raw/iris.csv", index=False)
