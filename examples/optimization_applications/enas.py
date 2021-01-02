import numpy as np
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Activation,
    Dropout,
)
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


"""
Efficient Neural Architecture Search via Parameter Sharing:
https://arxiv.org/pdf/1802.03268.pdf
"""

# to make the example quick
X_train = X_train[0:1000]
y_train = y_train[0:1000]

X_test = X_test[0:1000]
y_test = y_test[0:1000]

print("X_train.shape[1:]", X_train.shape)


model_pretrained = Sequential()
model_pretrained.add(
    Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:])
)
model_pretrained.add(Activation("relu"))
model_pretrained.add(Conv2D(32, (3, 3)))
model_pretrained.add(Activation("relu"))
model_pretrained.add(MaxPooling2D(pool_size=(2, 2)))
model_pretrained.add(Dropout(0.25))

model_pretrained.add(Conv2D(32, (3, 3), padding="same"))
model_pretrained.add(Activation("relu"))
model_pretrained.add(Dropout(0.25))

model_pretrained.add(Flatten())
model_pretrained.add(Dense(200))
model_pretrained.add(Activation("relu"))
model_pretrained.add(Dropout(0.5))
model_pretrained.add(Dense(10))
model_pretrained.add(Activation("softmax"))

model_pretrained.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model_pretrained.fit(X_train, y_train, epochs=5, batch_size=256)


print(model_pretrained.summary())


def cnn(opt):
    model = model_pretrained
    n_layers = len(model.layers)

    for i in range(n_layers - 9):
        model.pop()

    for layer in model.layers:
        layer.trainable = False

    model = opt["conv_layer.0"](model)
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(opt["neurons.0"]))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=5, batch_size=256)

    _, score = model.evaluate(x=X_test, y=y_test)

    return score


def conv1(model):
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model


def conv2(model):
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    return model


def conv3(model):
    return model


search_space = {
    "conv_layer.0": [conv1, conv2, conv3],
    "neurons.0": list(range(100, 1000, 100)),
}


hyper = Hyperactive()
hyper.add_search(cnn, search_space, n_iter=5)
hyper.run()
