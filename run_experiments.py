from src.data.pipeline import Pipeline

DATA_PATH = "data/raw/iris.csv"
EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "outputs/metrics"
TARGET = "species"

if __name__ == "__main__":
    pipeline = Pipeline(DATA_PATH, TARGET, EXPERIMENTS_DIR, OUTPUT_DIR)
    pipeline.run(scale=True)
