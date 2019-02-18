import tensorflow as tf
import logging
import utils.tf_variables as tf_variables
from .base_model import BaseModel
from .base_ops import *


def two_layer(x, num_classes=10, dropout_keep_prob=0.5, hidden_layer=256):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28 * 28])

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([28 * 28, hidden_layer])
        b_fc1 = bias_variable([hidden_layer])
        h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([hidden_layer, num_classes])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv, None


# Two-Layer Fully-Connected Network
class FCN(BaseModel):
    def __init__(self, dataset, batch_size=-1, learning_rate=1e-2, num_classes=10):
        super(FCN).__init__()
        self.dataset = dataset
        # probability of drop
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(
            tf.float32, [None, dataset.width, dataset.height, dataset.channels])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])
        self.lrn_rate = learning_rate
        self.y_conv, self.endpoint = two_layer(
            self.x, num_classes=num_classes, dropout_keep_prob=self.keep_prob)
        logging.debug("shape of output: {}".format(self.y_conv))

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=self.y_conv)
        self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
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
