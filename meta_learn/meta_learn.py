import os

from .collect_data import DataCollector
from .meta_regressor import MetaRegressor
from .search import Search


class MetaLearn:
    def __init__(self, search_config, metric="accuracy"):

        self.search_config = search_config
        self.metric = metric

        current_path = os.path.realpath(__file__)
        self.meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = self.meta_learn_path + "/meta_data/"
        self.meta_regressor_path = self.meta_learn_path + "/meta_regressor/"

    def search(self, data_config):
        search = Search(self.search_config)
        model, best_score = search.search_optimum(data_config)

        print("best_score", best_score)
        print("model", model)

        return model

    def score(self):
        pass

    def true_best_score(self, X_train, y_train):
        pass

    def meta_score(self, X_train, y_train):
        pass

    def fit(self, train_meta):
        pass

    def extract(self, dataset_dict):
        pass
