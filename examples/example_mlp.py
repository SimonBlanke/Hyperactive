from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from hyperactive import RandomAnnealingOptimizer

breast_cancer_data = load_breast_cancer()

X = breast_cancer_data.data
y = breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [3], "batch_size": [100], "verbose": [0]},
    "keras.layers.Dense.1": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.2": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.3": {"units": [1], "activation": ["softmax"]},
}

Optimizer = RandomAnnealingOptimizer(search_config, n_iter=3, metric="accuracy")
# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)
# predict from test data
prediction = Optimizer.predict(X_test)
# calculate score
score = Optimizer.score(X_test, y_test)
