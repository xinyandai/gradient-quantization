# Most of the tensorflow code is adapted from Tensorflow's tutorial on using
# CNNs to train MNIST
# https://www.tensorflow.org/get_started/mnist/pros#build-a-multilayer-convolutional-network.  # noqa: E501

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging

import tf_variables

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MODELC_VARIABLES = 'modelc_variables'
UPDATE_OPS_COLLECTION = 'modelc_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

activation = tf.nn.relu


def inference(x, is_training, use_bias=True, num_classes=10, keep_prob=0.5):
    c = dict()
    c['is_training'] = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_classes'] = num_classes
    c['ksize'] = 3
    c['stride'] = 1
    c['conv_filters_out'] = 96
    with tf.variable_scope('layer1'):
        x = conv(x, c)
        x = activation(x)
    with tf.variable_scope('layer2'):
        x = conv(x,c)
        x = activation(x)
    with tf.variable_scope('layer3'):
        c['stride'] = 2
        x = conv(x,c)
        x = activation(x)
        x = tf.layers.dropout(x, keep_prob)
    with tf.variable_scope('layer4'):
        c['conv_filters_out'] = 192
        x = conv(x,c)
        x = activation(x)
    with tf.variable_scope('layer5'):
        c['conv_filters_out'] = 192
        # layer 5
        x = conv(x, c)
        x = activation(x)
    with tf.variable_scope('layer6'):
        c['conv_filters_out'] = 192
        c['stride'] = 2
        x = conv(x, c)
        x = activation(x)
        x = tf.layers.dropout(x, keep_prob)
    with tf.variable_scope('layer7'):
        c['conv_filters_out'] = 192
        x = conv(x, c)
        x = activation(x)
    with tf.variable_scope('layer8'):
        c['ksize'] = 1
        c['conv_filters_out'] = 192
        x = conv(x, c)
        x = activation(x)
    with tf.variable_scope('layer9'):
        c['ksize'] = 1
        c['conv_filters_out'] = 10
        x = conv(x, c)
        x = activation(x)
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    return x


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)
    return loss_


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, MODELC_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                        shape=shape,
                        dtype='float',
                        initializer=initializer,
                        weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def maxpool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


class ModelC(object):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        # probability of drop
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [batch_size, dataset.width, dataset.height, dataset.channels])
        self.y_ = tf.placeholder(tf.float32, [batch_size, num_classes])

        boundaries = [int(10000 * i) for i in range(2, 4)]
        values = [0.01, 0.001, 0.0001]
        self.lrn_rate = tf.train.piecewise_constant(self.global_step,
                                                    boundaries, values)
        tf.summary.scalar('learning rate', self.lrn_rate)

        # Build the graph for the deep net
        self.logits = inference(self.x, is_training=True, num_classes=num_classes, keep_prob=self.keep_prob)

        with tf.name_scope('loss'):
            self.cost = loss(self.logits, self.y_)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            self.train_step = self.optimizer.minimize(self.cost)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1),
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
            self.cost, self.sess)
        self.grads = self.optimizer.compute_gradients(
            self.cost)
        self.grads_placeholder = [
            (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
            for grad in self.grads]
        self.apply_grads_placeholder = self.optimizer.apply_gradients(
            self.grads_placeholder, global_step=self.global_step)

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
        return self.sess.run([self.cost, self.accuracy],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})


if __name__ == '__main__':
    import mpi_dataset
    import numpy as np
    logging.basicConfig(level=logging.INFO)

    batch_size = 32
    test_batch_size = 1024

    dataset = mpi_dataset.download_cifar10_retry(0)
    net = ModelC(dataset, batch_size)

    i = 0
    from myutils import Timer

    timer = Timer()

    while True:
        # Compute and apply gradients.
        for _ in range(10):

            xs, ys = dataset.train.next_batch(batch_size)
            gradients = net.compute_gradients(xs, ys)
            net.apply_gradients(gradients)

            if i % 50 == 0:
                # Evaluate the current model.
                def test_loss_accuracy():
                    test_xs, test_ys = dataset.test.next_batch(batch_size)
                    loss, accuracy = net.compute_loss_accuracy(test_xs, test_ys)
                    return loss, accuracy

                def train_loss_accuracy():
                    valid_xs, valid_ys = dataset.valid.next_batch(batch_size)
                    valid_loss, valid_accuracy = net.compute_loss_accuracy(valid_xs, valid_ys)
                    return valid_loss, valid_accuracy

                test_accuracy = [
                    test_loss_accuracy() for _ in range(test_batch_size // batch_size + 1)
                ]
                train_accuracy = [
                    train_loss_accuracy() for _ in range(test_batch_size // batch_size + 1)
                ]
                ts_loss, ts_accuracy = np.mean(np.array(test_accuracy).reshape((-1, 2)), axis=0)
                tr_loss, tr_accuracy = np.mean(np.array(train_accuracy).reshape((-1, 2)), axis=0)
                print("%d, %.3f, %.3f, %.3f, %.3f, %.3f" % (i, timer.toc(), ts_loss, ts_accuracy, tr_loss, tr_accuracy))

            i += 1
