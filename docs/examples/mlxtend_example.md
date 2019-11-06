## Mlxtend

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"]
    )
    mlp = MLPClassifier(hidden_layer_sizes=para["hidden_layer_sizes"])
    svc = SVC(gamma="auto", probability=True)

    eclf = EnsembleVoteClassifier(
        clfs=[gbc, mlp, svc], weights=[2, 1, 1], voting="soft"
    )

    scores = cross_val_score(eclf, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(2, 12),
        "hidden_layer_sizes": (range(10, 100, 10),),
    }
}


opt = Hyperactive(search_config, n_iter=30)
opt.search(X, y)
```
