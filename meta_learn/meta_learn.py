import os

from importlib import import_module
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from .memory import Memory
from .meta_regressor import MetaRegressor
from .recognizer import Recognizer
from .predictor import Predictor


class MetaLearn:
    def __init__(self, search_config, metric="accuracy", cv=5):

        self.search_config = search_config
        self.metric = metric
        self.cv = cv

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        meta_data_path = meta_learn_path + "/meta_data/"
        meta_regressor_path = meta_learn_path + "/meta_regressor/"

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self.memory = Memory(self.search_config, meta_data_path=meta_data_path, cv=3)
        self.meta_regressor = MetaRegressor(meta_learn_path)
        self.recognizer = Recognizer(search_config)
        self.predictor = Predictor(search_config, meta_regressor_path)

        self.model_list = list(self.search_config.keys())

        self.model_name = self.model_list[0]

        self.model = self._import_model(self.model_name)

    def _import_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model

    def collect(self, dataset_config):
        self.memory.extract(dataset_config)

    def train(self, model_list):
        self.meta_regressor.train_meta_regressor(model_list)

    def search(self, dataset_config):
        X_test = self.recognizer.get_test_metadata(dataset_config)

        self.best_hyperpara_dict, self.best_score = self.predictor.search(X_test)

        # print("best_hyperpara_dict ", self.best_hyperpara_dict)
        # print("best_score          ", self.best_score)

    def _grid_search(self, X_train, y_train):

        search_dict = self.search_config[str(*self.search_config.keys())]

        self.model_grid_search = GridSearchCV(
            self.model(), search_dict, cv=self.cv, n_jobs=-1, verbose=0
        )
        self.model_grid_search.fit(X_train, y_train)

    def _grid_true_best_score(self, X_train, y_train):
        dict = {"min_samples_leaf": 9, "min_samples_split": 2, "n_estimators": 60}
        if dict:
            model_hyper = self.model(**dict)
            cv_results = cross_val_score(model_hyper, X_train, y_train, cv=self.cv)
            return cv_results.mean(), dict
        else:
            self._grid_search(X_train, y_train)
            para = self.model_grid_search.best_params_
            model_hyper = self.model(**para)
            cv_results = cross_val_score(model_hyper, X_train, y_train, cv=self.cv)
            return cv_results.mean(), para

    def _pred_prox_best_score(self):
        return self.best_score, self.best_hyperpara_dict

    def _pred_true_best_score(self, X_train, y_train):
        model_hyper = self.model(**self.best_hyperpara_dict)
        cv_results = cross_val_score(model_hyper, X_train, y_train, cv=self.cv)
        return cv_results.mean(), model_hyper.get_params()

    def _default_score(self, X_train, y_train):
        cv_results = cross_val_score(self.model(), X_train, y_train, cv=self.cv)
        return cv_results.mean(), self.model().get_params()

    def compare_scores(self, X_train, y_train):
        _default_score, _default_para = self._default_score(X_train, y_train)
        _pred_prox_best_score, _pred_prox_best_para = self._pred_prox_best_score()
        _pred_true_best_score, _pred_true_best_para = self._pred_true_best_score(
            X_train, y_train
        )
        _grid_true_best_score, _grid_search_para = self._grid_true_best_score(
            X_train, y_train
        )

        print("\n")
        print("_default_score         ", round(_default_score, 3))
        print("_pred_prox_best_score  ", round(_pred_prox_best_score, 3))
        print("_pred_true_best_score  ", round(_pred_true_best_score, 3))
        print("_grid_true_best_score  ", round(_grid_true_best_score, 3))
        print("\n")
        print("_default_para        ", _default_para)
        print("_pred_prox_best_para ", _pred_prox_best_para)
        print("_pred_true_best_para ", _pred_true_best_para)
        print("_grid_search_para    ", _grid_search_para)
