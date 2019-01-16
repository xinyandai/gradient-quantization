import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time

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

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")
activation = tf.nn.relu

def inference(x, is_training, use_bias=True, num_classes=10):
    c = Config()
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
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)
    return loss_

def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x

def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


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
    # weights = tf.get_variable(  name = "weights", 
    #                             shape = shape, 
    #                             dtype = 'float',
    #                             initializer = initializer, 
    #                             regularizer = tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY))
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def maxpool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
