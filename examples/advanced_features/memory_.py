from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from hyperactive import Hyperactive

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

def model(para, X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)

    return scores.mean(), model

search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

"""
The memory will remember previous evaluations done during the optimization process.
Instead of retraining the model, it accesses the memory and uses the saved score/loss.
This shows as a speed up during the optimization process, since the whole search space has been explored.
"""
opt = Hyperactive(search_config, n_iter=1000, memory=True)

# search best hyperparameter for given data
opt.fit(X, y)
