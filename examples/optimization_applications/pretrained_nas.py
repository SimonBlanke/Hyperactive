"""
This script describes how to save time during the optimization by
using a pretrained model. It is similar to the transer learning example,
but here you do the training and model creation of the pretrained model 
yourself.

The problem is that most of the optimization time is "waisted" by 
training the model. The time to find a new position to explore by
Hyperactive is very small compared to the training time of 
neural networks. This means, that we can do more optimization
if we keep the training time as little as possible. 

The idea of pretrained neural architecture search is to pretrain a complete model one time.
In the next step we remove the layers that should be optimized 
and make the remaining layers not-trainable.

This results in a partial, pretrained, not-trainable model that will be
used during the Hyperactive optimization.

You can now add layers to the partial model in the objective function
and add the parameters or layers that will be optimized by Hyperactive.

With each iteration of the optimization run we are only training 
the added layers of the model. This saves a lot of training time.

"""


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

# to make the example quick
X_train = X_train[0:1000]
y_train = y_train[0:1000]

X_test = X_test[0:1000]
y_test = y_test[0:1000]


# create model and train it
model_pretrained = Sequential()
model_pretrained.add(Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:]))
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


def cnn(opt):
    model = model_pretrained
    n_layers = len(model.layers)

    # delete the last 9 layers
    for i in range(n_layers - 9):
        model.pop()

    # set remaining layers to not-trainable
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


# conv 1, 2, 3 are functions that adds layers. We want to know which function is the best
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
