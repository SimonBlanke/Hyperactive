import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

class GPR0:
    def __init__(self):
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        
    def fit(self, X, y):
        self.m = GPy.models.GPRegression(X, y, self.kernel)
        self.m.optimize(messages=True)

    def predict(self, X):
        return self.m.predict(X)

class GPR1:
    def __init__(self):
        self.gpr = GaussianProcessRegressor(
                kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10
            )
        
    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X):
        return self.gpr.predict(X, return_std=True)


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=30, optimizer="Bayesian")


bayes_opt = {"Bayesian": {"gpr": GPR0()}}
opt = Hyperactive(X, y)
opt.search(search_config, n_iter=30, optimizer=bayes_opt)


bayes_opt = {"Bayesian": {"gpr": GPR1()}}
opt = Hyperactive(X, y)
opt.search(search_config, n_iter=30, optimizer=bayes_opt)
