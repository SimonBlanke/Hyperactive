import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from hyperactive import RandomSearch_Optimizer

(X_train, y_train), (X_test, y_test) = mnist.load_data()

size = 6000

X_train = X_train[0:size]
y_train = y_train[0:size]

X_train = X_train.reshape(size, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [10], "batch_size": [500], "verbose": [2]},
    "keras.layers.Conv2D.1": {
        "filters": [32, 64, 128],
        "kernel_size": range(3, 4),
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    "keras.layers.Conv2D.3": {
        "filters": [32, 64, 128],
        "kernel_size": [3],
        "activation": ["relu"],
        "input_shape": [(28, 28, 1)],
    },
    "keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    "keras.layers.Flatten.5": {},
    "keras.layers.Dense.6": {"units": range(30, 200, 10), "activation": ["softmax"]},
    "keras.layers.Dropout.7": {"rate": np.arange(0.4, 0.8, 0.1)},
    "keras.layers.Dense.8": {"units": [10], "activation": ["softmax"]},
}

Optimizer = RandomSearch_Optimizer(search_config, n_iter=10)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
