from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# to make the example quick
X_train = X_train[0:1000]
y_train = y_train[0:1000]

X_test = X_test[0:1000]
y_test = y_test[0:1000]


def cnn(para, X_train, y_train):
    nn = Sequential()
    nn.add(
        Conv2D(para["filter.0"], (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    nn.add(Activation("relu"))
    nn.add(Conv2D(para["filter.0"], (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Conv2D(para["filter.0"], (3, 3), padding="same"))
    nn.add(Activation("relu"))
    nn.add(Conv2D(para["filter.0"], (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Flatten())
    nn.add(Dense(para["layer.0"]))
    nn.add(Activation("relu"))
    nn.add(Dropout(0.5))
    nn.add(Dense(10))
    nn.add(Activation("softmax"))

    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    nn.fit(X_train, y_train, epochs=10, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_space = {
    "filter.0": [16, 32, 64, 128],
    "layer.0": list(range(100, 1000, 100)),
}


X_train = np.asarray(X_train, order="C")
y_train = np.asarray(y_train, order="C")

hyper = Hyperactive(X_train, y_train)
hyper.add_search(cnn, search_space, n_iter=5)
hyper.run()

