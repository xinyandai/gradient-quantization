# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

import tf_variables


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


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class LinearRegression(object):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        with tf.Graph().as_default():
            # probability of drop
            self.keep_prob = tf.placeholder(tf.float32)
            self.x = tf.placeholder(tf.float32, [None, dataset.width, dataset.height, dataset.channels])
            self.y_ = tf.placeholder(tf.float32, [None, num_classes])
            # Build the graph for the deep net
            self.y_conv, self.endpoint = learning_regression(
                self.x, num_classes=num_classes, dropout_keep_prob=self.keep_prob)
            logging.debug("shape of output: {}".format(self.y_conv))

            with tf.name_scope('loss'):
                cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.y_conv, labels=self.y_))
            self.cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy)

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

    def compute_loss_accuracy(self, x, y):
        return self.sess.run([self.cross_entropy, self.accuracy],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})
