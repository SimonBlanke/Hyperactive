## Tensorflow

```python
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from hyperactive import Hyperactive

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

learning_rate = 0.001
num_steps = 500
batch_size = 128

num_input = 784
num_classes = 10
dropout = 0.25

X_train = mnist.train.images
y_train = mnist.train.labels

X_test = mnist.test.images
y_test = mnist.test.labels


def cnn(para, X_train, y_train):
    def conv_net(x_dict, n_classes, dropout, reuse, is_training):
        with tf.variable_scope("ConvNet", reuse=reuse):
            x = x_dict["images"]
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(x, para["filters_0"], 5, activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(conv1, para["filters_1"], 3, activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(fc1, para["dense_0"])
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            out = tf.layers.dense(fc1, n_classes)

        return out

    def model_fn(features, labels, mode):
        logits_train = conv_net(
            features, num_classes, dropout, reuse=False, is_training=True
        )
        logits_test = conv_net(
            features, num_classes, dropout, reuse=True, is_training=False
        )

        pred_classes = tf.argmax(logits_test, axis=1)
        # pred_probas = tf.nn.softmax(logits_test)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)
            )
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={"accuracy": acc_op},
        )

        return estim_specs

    model = tf.estimator.Estimator(model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": X_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True,
    )
    model.train(input_fn, steps=num_steps)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": X_test}, y=y_test, batch_size=batch_size, shuffle=False
    )
    e = model.evaluate(input_fn)

    return float(e["accuracy"])


search_config = {
    cnn: {
        "filters_0": [16, 32, 64],
        "filters_1": [16, 32, 64],
        "dense_0": range(100, 2000, 100),
    }
}

opt = Hyperactive(X_train, y_train)
opt.search(search_config, n_iter=20)
```

## Keras

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

from hyperactive import Hyperactive

import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


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
    nn.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_config = {cnn: {"filter.0": [16, 32, 64, 128], "layer.0": range(100, 1000, 100)}}


X_train = np.asarray(X_train, order="C")
y_train = np.asarray(y_train, order="C")

opt = Hyperactive(X_train, y_train)
opt.search(search_config, n_iter=5)
```

