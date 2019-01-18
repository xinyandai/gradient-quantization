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
    def __init__(self, x, y, num_classes=-1, seed=0, one_hot=True, distort=False):
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

        # shuffle
        np.random.seed(seed)
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)
        self.x[:] = self.x[self.idx]
        self.y[:] = self.y[self.idx]

        _, self.width, self.height, self.channels = np.shape(self.x)
        self.distort = distort
        self.distort_sess = tf.Session()
        self.x_origin = self.x.copy()
        # standardization
        self.x = self.distort_sess.run(
            self._distort_ops(distort=False),
            feed_dict={self.image: self.x_origin})
        # distort
        if self.distort:
            self.distorted_image = self._distort_ops(distort=self.distort)

    def distort_images(self):
        self.x = self.distort_sess.run(self.distorted_image, feed_dict={self.image: self.x_origin})

    def _distort_ops(self, distort):
        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        self.image = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels])
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.map_fn(
            lambda img: tf.random_crop(img, [self.height, self.width, self.channels]), self.image)

        if distort:
            # Randomly flip the image horizontally.
            distorted_image = tf.map_fn(
                lambda img: tf.image.random_flip_left_right(img), distorted_image)
            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            distorted_image = tf.map_fn(
                lambda img: tf.image.random_brightness(img, max_delta=63), distorted_image
            )
            distorted_image = tf.map_fn(
                lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), distorted_image
            )
            # Subtract off the mean and divide by the variance of the pixels.
            # float_image = tf.image.per_image_whitening(distorted_image)

        float_image = tf.map_fn(
            lambda img: tf.image.per_image_standardization(img), distorted_image
        )

        return float_image

    def next_batch(self, batch_size):
        assert batch_size > 0
        batch = (self.x[self.i:self.i + batch_size, :], self.y[self.i: self.i + batch_size, :])
        if self.i + batch_size >= self.size:
            if self.distort:
                self.distort_images()
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
        BatchDataset(np.expand_dims(x_train, axis=3), y_train, num_classes, seed, distort=False),
        BatchDataset(np.expand_dims(x_train, axis=3), y_train, num_classes, seed, distort=False),
        BatchDataset(np.expand_dims(x_test, axis=3), y_test, num_classes, seed, distort=False)
    )


def download_cifar10_retry(seed):
    num_classes = 10
    # (50000, 32, 32, 3)  (50000, 1)
    # (10000, 32, 32, 3)  (10000, 1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return Datasets(
        BatchDataset(x_train, y_train.flatten(), num_classes, seed, distort=True),
        BatchDataset(x_train, y_train.flatten(), num_classes, seed, distort=False),
        BatchDataset(x_test, y_test.flatten(), num_classes, seed, distort=False)
    )