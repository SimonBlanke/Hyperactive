import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

X_list, y_list = [X], [y]

def data_aug(X, y, sample_multi=5, feature_multi=5):
    X_list, y_list = [], []

    n_samples = X.shape[0]
    n_features = X.shape[1]

    for sample in range(1, sample_multi+1):
        idx_sample = np.random.randint(n_samples, size=int(n_samples / sample))

        for feature in range(1, feature_multi+1):
            idx_feature = np.random.randint(n_features, size=int(n_features / feature))

            X_temp_ = X[idx_sample, :]
            X_temp = X_temp_[:, idx_feature]
            y_temp = y[idx_sample]

            X_list.append(X_temp)
            y_list.append(y_temp)

    return X_list, y_list


def meta_opt(para, X_list, y_list):
    scores = []
    
    for X, y in zip(X_list, y_list):
        X_list, y_list = data_aug(X, y, sample_multi=3, feature_multi=3)

        for X, y in zip(X_list, y_list):
            def model(para, X, y):
                model = DecisionTreeClassifier(
                    max_depth=para["max_depth"],
                    min_samples_split=para["min_samples_split"],
                    min_samples_leaf=para["min_samples_leaf"],
                )
                scores = cross_val_score(model, X, y, cv=3)

                return scores.mean()

            search_config = {
                model: {
                    "max_depth": range(2, 50),
                    "min_samples_split": range(2, 50),
                    "min_samples_leaf": range(1, 50),
                }
            }

            opt = Hyperactive(
                search_config,
                optimizer={
                    "ParticleSwarm": {"inertia": para["inertia"], "cognitive_weight": para["cognitive_weight"], "social_weight": para["social_weight"]}
                }, 
                n_iter=30,
                verbosity=None,
            )
            opt.search(X, y)
            score = opt.score_best
            scores.append(score)

    return np.array(scores).mean()


search_config = {
    meta_opt: {
        "inertia": np.arange(0, 1, 0.01),
        "cognitive_weight": np.arange(0, 1, 0.01),
        "social_weight": np.arange(0, 1, 0.01),
    }
}

opt = Hyperactive(search_config, optimizer="Bayesian", n_iter=30)
opt.search(X_list, y_list)
