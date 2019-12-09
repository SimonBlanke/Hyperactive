from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def cnn(para, X_train, y_train):
    model = Sequential()
    model.add(
        Conv2D(para["filter.0"], (3, 3), padding="same", input_shape=X_train.shape[1:])
    )
    model.add(Activation("relu"))
    model.add(Conv2D(para["filter.0"], (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(para["filter.0"], (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(para["filter.0"], (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(para["layer.0"]))
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


search_config = {cnn: {"filter.0": [16, 32, 64, 128], "layer.0": range(100, 1000, 100)}}

opt = Hyperactive(search_config, n_iter=5)
opt.search(X_train, y_train)
