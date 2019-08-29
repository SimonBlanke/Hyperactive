from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import SimulatedAnnealingOptimizer

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [1], "batch_size": [300], "verbose": [0]},
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
    "keras.layers.Dense.4": {"units": [10], "activation": ["sigmoid"]},
}


opt = SimulatedAnnealingOptimizer(search_config, n_iter=3, warm_start=False)

# search best hyperparameter for given data
opt.fit(X_train, y_train)

# predict from test data
prediction = opt.predict(X_test)

# calculate score
score = opt.score(X_test, y_test)
