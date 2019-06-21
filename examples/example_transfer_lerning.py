from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import SimulatedAnnealing_Optimizer

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [1], "batch_size": [100]},
    # just add the pretrained model as a layer like this:
    "keras.applications.MobileNet.1": {
        "weights": ["imagenet"],
        "input_shape": [(32, 32, 3)],
        "include_top": [False],
    },
    "keras.layers.Flatten.2": {},
    "keras.layers.Dense.3": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.4": {
        "units": range(5, 15),
        "activation": ["relu"],
        "kernel_initializer": ["uniform"],
    },
    "keras.layers.Dense.5": {"units": [10], "activation": ["sigmoid"]},
}


Optimizer = SimulatedAnnealing_Optimizer(search_config, n_iter=1, warm_start=False)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
