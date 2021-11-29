"""
This example shows how you can search for the best models in each layer in a 
stacking ensemble. 

We want to create a stacking ensemble with 3 layers:
    - a top layer with one model
    - a middle layer with multiple models
    - a bottom layer with multiple models

We also want to know how many models should be used in the middle and bottom layer.
For that we can use the helper function "get_combinations". It works as follows:

input = [1, 2 , 3]
output = get_combinations(input, comb_len=2)
output: [[1, 2], [1, 3], [2, 3], [1, 2, 3]]

Instead of numbers we insert models into "input". This way we get each combination
with more than 2 elements. Only 1 model per layer would not make much sense.

The ensemble itself is created via the package "mlxtend" in the objective-function "stacking".
"""

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
from sklearn.linear_model import RidgeClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

# define models that are used in search space
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
    lvl_1_ = opt["lvl_1"]()
    lvl_0_ = opt["lvl_0"]()
    top_ = opt["top"]()

    stack_lvl_0 = StackingClassifier(classifiers=lvl_0_, meta_classifier=top_)
    stack_lvl_1 = StackingClassifier(classifiers=lvl_1_, meta_classifier=stack_lvl_0)
    scores = cross_val_score(stack_lvl_1, X, y, cv=3)

    return scores.mean()


# helper function to create search space dimensions
def get_combinations(models, comb_len=2):
    def _list_in_list_of_lists(list_, list_of_lists):
        for list__ in list_of_lists:
            if set(list_) == set(list__):
                return True

    comb_list = []
    for i in range(0, len(models) + 1):
        for subset in itertools.permutations(models, i):
            if len(subset) < comb_len:
                continue
            if _list_in_list_of_lists(subset, comb_list):
                continue

            comb_list.append(list(subset))

    comb_list_f = []
    for comb_ in comb_list:

        def _func_():
            return comb_

        _func_.__name__ = str(i) + "___" + str(comb_)
        comb_list_f.append(_func_)

    return comb_list_f


def lr_f():
    return lr


def dtc_f():
    return dtc


def gnb_f():
    return gnb


def rc_f():
    return rc


models_0 = [gpc, dtc, mlp, gnb, knn]
models_1 = [gbc, rfc, etc]

stack_lvl_0_clfs = get_combinations(models_0)
stack_lvl_1_clfs = get_combinations(models_1)


print("\n stack_lvl_0_clfs \n", stack_lvl_0_clfs, "\n")


search_space = {
    "lvl_1": stack_lvl_1_clfs,
    "lvl_0": stack_lvl_0_clfs,
    "top": [lr_f, dtc_f, gnb_f, rc_f],
}

"""
hyper = Hyperactive()
hyper.add_search(stacking, search_space, n_iter=3)
hyper.run()
"""
