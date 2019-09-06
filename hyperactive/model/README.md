# Supported Packages



## Machine Learning

#### Scikit-learn


<br>

#### XGBoost


<br>

#### LightGBM


<br>

#### CatBoost


<br>

## Deep Learning


<br>

#### Keras




## Distribution

#### [Multiprocessing](https://github.com/SimonBlanke/Hyperactive/blob/master/examples/example_multiprocessing.py)
The multiprocessing in hyperactive works by creating additional searches, that run in parallel without any shared memory. This provides the possibility of hyperparameter-tuning of different models at the same time. If one single model should be tuned as fast as possible n_jobs in the optimizer should be set to '1', while n_jobs (of the model) in the search_config should be set to '-1'.

<details><summary>Two searches with eight cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "n_jobs": [4],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "n_jobs": [4],
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=2, verbosity=0)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
```

</p>
</details>


<details><summary>One search with all cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "n_jobs": [-1],
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=1, verbosity=0)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
```

</p>
</details>




<details><summary>Multiple searches with all cpu-cores:</summary>
<p>

```python
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 10),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(10, 100, 10),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=300, n_jobs=-1, verbosity=0)
```

</p>
</details>
