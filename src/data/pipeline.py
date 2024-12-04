import os

from src.data.data_processor import DataProcessor
from src.data.experiment import Experiment


class Pipeline:
    def __init__(self, data_path, target, experiments_dir, output_dir):
        self.data_processor = DataProcessor(data_path, target)
        self.experiments_dir = experiments_dir
        self.output_dir = output_dir

    def run(self, scale=False):
        X_train, X_test, y_train, y_test = self.data_processor.preprocess(scale=scale)

        for config_file in os.listdir(self.experiments_dir):
            if config_file.endswith(".yaml"):
                config_path = os.path.join(self.experiments_dir, config_file)
                experiment = Experiment(config_path, self.output_dir)

                experiment.train(X_train, y_train)
                metrics = experiment.evaluate(X_test, y_test)
                experiment.save_model(config_file.split(".")[0])
