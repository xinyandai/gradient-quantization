from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Datasets(object):
    def __init__(self, train, valid, test):
        self.train = train
        self.test = test
        self.valid = valid
        _, self.width, self.height, self.channels = np.shape(self.train.x)


class BatchDataset(object):
    def __init__(self, x, y, num_classes=-1, seed=0, one_hot=True):
        assert len(x) == len(y)

        self.i = 0
        self.size = len(x)

        self.x = x.copy()

        if one_hot:
            if num_classes == -1:
                num_classes = np.max(y) + 1
            self.y = np.eye(num_classes, dtype=np.float32)[y]
        else:
            self.y = y.copy()

        np.random.seed(seed)

        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)

        self.x[:] = self.x[self.idx]
        self.y[:] = self.y[self.idx]

    def next_batch(self, batch_size):
        assert batch_size > 0
        batch = (self.x[self.i:self.i + batch_size, :], self.y[self.i: self.i + batch_size, :])
        if self.i + batch_size >= self.size:
            self.i = batch_size + self.i - self.size
            padding = (self.x[0:self.i, :], self.y[0: self.i, :])
            return (
                np.concatenate((batch[0], padding[0]), axis=0),
                np.concatenate((batch[1], padding[1]), axis=0)
            )
        else:
            self.i += batch_size
            return batch


def download_mnist_retry(seed):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return Datasets(
        BatchDataset(np.expand_dims(x_train, axis=3), y_train, num_classes, seed),
        BatchDataset(np.expand_dims(x_train, axis=3), y_train, num_classes, seed),
        BatchDataset(np.expand_dims(x_test, axis=3), y_test, num_classes, seed)
    )


def download_cifar10_retry(seed):
    num_classes = 10
    # (50000, 32, 32, 3)  (50000, 1)
    # (10000, 32, 32, 3)  (10000, 1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return Datasets(
        BatchDataset(x_train, y_train.flatten(), num_classes, seed),
        BatchDataset(x_train, y_train.flatten(), num_classes, seed),
        BatchDataset(x_test, y_test.flatten(), num_classes, seed)
    )