from sklearn import svm, datasets

from hyperactive.integrations import HyperactiveSearchCV
from hyperactive.optimizers import RandomSearchOptimizer

iris = datasets.load_iris()


svc = svm.SVC()
opt = RandomSearchOptimizer()
parameters = {"kernel": ["linear", "rbf"], "C": [1, 10]}

search = HyperactiveSearchCV(svc, opt, parameters)
search.fit(iris.data, iris.target)

print("\n search.get_params() \n", search.get_params())
