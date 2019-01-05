from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tf_variables

import tensorflow as tf
import numpy as np

from model_simplenn import deepnn


class Datasets(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test
        _, self.width, self.height, self.channels = np.shape(self.train.x)


class BatchDataset(object):
    def __init__(self, x, y, num_classes=-1, seed=0, one_hot=True):
        assert len(x) == len(y)

        self.i = 0
        self.size = len(x)

        self.x = x

        if one_hot:
            if num_classes == -1:
                num_classes = np.max(y) + 1
            self.y = np.eye(num_classes, dtype=np.float32)[y]
        else:
            self.y = y

        np.random.seed(seed)

        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)

        self.x[:] = self.x[self.idx]
        self.y[:] = self.y[self.idx]

    def next_batch(self, batch_size):
        assert batch_size > 0
        batch = (self.x[self.i:self.i + batch_size, :], self.y[self.i: self.i + batch_size, :])
        if self.i + batch_size >= self.size:
            self.i = 0
        else:
            self.i += batch_size
        return batch


def download_mnist_retry(seed):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return Datasets(
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
        BatchDataset(x_test, y_test.flatten(), num_classes, seed)
    )


class ModelCNN(object):
    def __init__(self, dataset, net, learning_rate=1e-2, num_classes=10):
        with tf.Graph().as_default():

            # probability of drop
            self.keep_prob = tf.placeholder(tf.float32)
            self.x = tf.placeholder(tf.float32, [None, dataset.width, dataset.height, dataset.channels])
            self.y_ = tf.placeholder(tf.float32, [None, num_classes])
            # Build the graph for the deep net
            if net == deepnn:
                images = tf.image.resize_images(self.x, [28, 28])
                if self.x.shape[-1].value > 1:
                    images = tf.image.rgb_to_grayscale(images, name=None)
            else:
                images = tf.image.resize_images(self.x, [224, 224])
                if self.x.shape[-1].value == 1:
                    images = tf.image.grayscale_to_rgb(images, name=None)

            self.y_conv, self.endpoint = net(
                images, num_classes=num_classes, dropout_keep_prob=self.keep_prob)

            logging.debug("shape of output: {}".format(self.y_conv))

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_, logits=self.y_conv)
            self.cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('adam_optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(
                    self.cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.y_conv, 1),
                                              tf.argmax(self.y_, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1,
                    device_count={'GPU': 0})
            )
            self.sess.run(tf.global_variables_initializer())

            # Helper values.
            self.variables = tf_variables.TensorFlowVariables(
                self.cross_entropy, self.sess)
            self.grads = self.optimizer.compute_gradients(
                self.cross_entropy)
            self.grads_placeholder = [
                (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
                for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 0.5})

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(self.accuracy,
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})

    def compute_loss_accuracy(self, x, y):
        return self.sess.run([self.cross_entropy, self.accuracy],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})

