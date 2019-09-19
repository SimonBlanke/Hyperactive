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

# Without scatter initialization
opt = SimulatedAnnealingOptimizer(
    search_config, optimizer="HillClimbing", n_iter=20, random_state=0, scatter_init=False
)
opt.fit(X, y)


# With scatter initialization
opt = Hyperactive(
    search_config, optimizer="HillClimbing", n_iter=20, random_state=0, scatter_init=True
)
opt.fit(X, y)
