import itertools


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.ridge import RidgeClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
etc = ExtraTreesClassifier()

mlp = MLPClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()

lr = LogisticRegression()
rc = RidgeClassifier()


def stacking(opt):
    stack_lvl_0 = StackingClassifier(
        classifiers=opt["lvl_0"], meta_classifier=opt["top"]
    )
    stack_lvl_1 = StackingClassifier(
        classifiers=opt["lvl_1"], meta_classifier=stack_lvl_0
    )
    scores = cross_val_score(stack_lvl_1, X, y, cv=3)

    return scores.mean()


def get_combinations(models):
    comb = []
    for i in range(0, len(models) + 1):
        for subset in itertools.permutations(models, i):
            if len(subset) == 0:
                continue
            comb.append(list(subset))
    return comb


top = [lr, dtc, gnb, rc]
models_0 = [gpc, dtc, mlp, gnb, knn]
models_1 = [gbc, rfc, etc]

stack_lvl_0_clfs = get_combinations(models_0)
stack_lvl_1_clfs = get_combinations(models_1)


search_space = {
    "lvl_1": stack_lvl_1_clfs,
    "lvl_0": stack_lvl_0_clfs,
    "top": top,
}
hyper = Hyperactive()
hyper.add_search(stacking, search_space, n_iter=20)
hyper.run()

