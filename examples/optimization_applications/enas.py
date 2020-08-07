import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
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


def conv1(model):
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model


def conv2(model):
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    return model


def conv3(model):
    return model


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
model_pretrained.fit(X_train, y_train, epochs=50, batch_size=128)

n_layers = len(model_pretrained.layers)

for i in range(n_layers - 8):
    model_pretrained.pop()

for layer in model_pretrained.layers:
    layer.trainable = False

print(model_pretrained.summary())


def cnn(para, X_train, y_train):
    """
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    """
    model = para["model_pretrained"]

    model = para["conv_layer.0"](model)
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(para["neurons.0"]))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = model.evaluate(x=X_test, y=y_test)

    return score


search_space = {
    "model_pretrained": [model_pretrained],
    "conv_layer.0": [conv1, conv2, conv3],
    "neurons.0": list(range(100, 1000, 100)),
}

# make numpy array "C-contiguous". This is important for saving meta-data
X_train = np.asarray(X_train, order="C")
y_train = np.asarray(y_train, order="C")

hyper = Hyperactive(X_train, y_train)
hyper.add_search(cnn, search_space, n_iter=5)
hyper.run()
