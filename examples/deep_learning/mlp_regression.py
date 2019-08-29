from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from hyperactive import RandomAnnealingOptimizer

boston_data = load_boston()

X = boston_data.data
y = boston_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [5], "batch_size": [200], "verbose": [1]},
    "keras.layers.Dense.1": {
        "units": range(5, 100),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.2": {"units": [1], "activation": ["linear"]},
}

Optimizer = RandomAnnealingOptimizer(
    search_config, n_iter=20, metric="mean_squared_error"
)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate score
score = Optimizer.score(X_test, y_test)

print("\ntest score of best model:", score)
