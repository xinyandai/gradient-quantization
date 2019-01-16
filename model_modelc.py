# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

import tf_variables


def _get_variable(name,
                  shape,
                  initializer,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"

    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           trainable=trainable)


def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    weights = tf.get_variable('weights', shape=shape, dtype='float')

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def model_conv(x, num_classes=10):
    ksize = 3
    stride = 1
    conv_filters_out = 96
    activation = tf.nn.relu

    with tf.variable_scope('layer1'):
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer2'):
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer3'):
        stride = 2
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer4'):
        conv_filters_out = 192
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer5'):
        conv_filters_out = 192
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer6'):
        conv_filters_out = 192
        stride = 2
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer7'):
        conv_filters_out = 192
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer8'):
        ksize = 1
        conv_filters_out = 192
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    with tf.variable_scope('layer9'):
        ksize = 1
        conv_filters_out = 10
        x = conv(x, ksize, stride, conv_filters_out)
        x = activation(x)

    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    return x, None


class ModelC(object):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        with tf.Graph().as_default():
            # probability of drop
            self.keep_prob = tf.placeholder(tf.float32)
            self.x = tf.placeholder(tf.float32, [batch_size, dataset.width, dataset.height, dataset.channels])
            self.y_ = tf.placeholder(tf.float32, [batch_size, num_classes])
            # Build the graph for the deep net

            self.y_conv, self.endpoint = model_conv(self.x, num_classes=num_classes)

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_, logits=self.y_conv)
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
                    device_count={'GPU': 1})
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
