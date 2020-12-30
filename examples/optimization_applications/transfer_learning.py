import numpy as np
from keras.models import Sequential
from keras import applications
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

nn = applications.VGG19(weights="imagenet", include_top=False)

for layer in nn.layers[:5]:
    layer.trainable = False


def cnn(opt):
    nn = Sequential()

    nn.add(Flatten())
    nn.add(Dense(opt["Dense.0"]))
    nn.add(Activation("relu"))
    nn.add(Dropout(opt["Dropout.0"]))
    nn.add(Dense(10))
    nn.add(Activation("softmax"))

    nn.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    nn.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_space = {
    "Dense.0": list(range(100, 1000, 100)),
    "Dropout.0": list(np.arange(0.1, 0.9, 0.1)),
}


hyper = Hyperactive()
hyper.add_search(cnn, search_space, n_iter=5)
hyper.run()
