import tensorflow as tf
import logging
import utils.tf_variables as tf_variables
from .base_model import BaseModel
from .base_ops import *


def deepnn(x, num_classes=10, dropout_keep_prob=0.5):
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
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout_keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, None


class CNN(BaseModel):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        self.dataset = dataset
        # probability of drop
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(
            tf.float32, [None, dataset.width, dataset.height, dataset.channels])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])
        # Build the graph for the deep net
        self.y_conv, self.endpoint = deepnn(
            self.x, num_classes=num_classes, dropout_keep_prob=self.keep_prob)
        logging.debug("shape of output: {}".format(self.y_conv))

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y_conv)
        self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

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
        self.add_helper_vars()
        self.sess.run(tf.global_variables_initializer())
