# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

import utils.tf_variables as tf_variables
from .base_model import BaseModel
from .base_ops import weight_variable, bias_variable


def learning_regression(x, num_classes=10, dropout_keep_prob=0.5):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
            the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with
            values equal to the logits of classifying the digit into one of 10
            classes (the digits 0-9). keep_prob is a scalar placeholder for the
            probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28 * 28])

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc = weight_variable([28 * 28, num_classes])
        b_fc = bias_variable([num_classes])

        y_conv = tf.matmul(x_image, W_fc) + b_fc
    return y_conv, None


class LinearRegression(BaseModel):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(
            tf.float32, [None, dataset.width, dataset.height, dataset.channels])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])
        # Build the graph for the deep net
        self.y_conv, self.endpoint = learning_regression(
            self.x, num_classes=num_classes, dropout_keep_prob=self.keep_prob)
        logging.debug("shape of output: {}".format(self.y_conv))

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.y_conv, labels=self.y_))
        self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1),
                                          tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.add_helper_vars()

        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                device_count={'GPU': 0})
        )
        self.sess.run(tf.global_variables_initializer())
