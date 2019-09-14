# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from hyperactive import Hyperactive, Hydra, Iota, Insight, MetaLearn
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target


def test_classes():
    insight = Insight(X, y)
    hydra = Hydra()
    iota = Iota()
