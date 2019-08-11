# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

search_config = {
    "sklearn.ensemble.GradientBoostingClassifier": {"n_estimators": range(1, 8, 2)}
}


def test_metalearn():
    from hyperactive import MetaLearn

    ml = MetaLearn(search_config)
    ml.collect(X, y)
    ml.train()
    ml.search(X, y)
