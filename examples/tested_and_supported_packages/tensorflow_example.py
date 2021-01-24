import tensorflow as tf

from hyperactive import Hyperactive

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def cnn(params):
    nn = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    nn.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    nn.fit(x_train, y_train, epochs=5)
    _, score = nn.evaluate(x=x_test, y=y_test)

    return score


search_space = {
    "filters_0": [16, 32, 64],
    "filters_1": [16, 32, 64],
    "dense_0": list(range(100, 2000, 100)),
}

hyper = Hyperactive()
hyper.add_search(cnn, search_space, n_iter=5)
hyper.run()
