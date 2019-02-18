import tensorflow as tf
from typing import Tuple, List
import tf_variables


class BaseModel(object):
    def __init__(self):
        self.sess: tf.Session
        self.grads: tf.Tensor
        self.grads_placeholder: tf.Tensor
        self.apply_grads_placeholder: tf.Tensor
        self.x: tf.Tensor
        self.y_: tf.Tensor
        self.keep_prob: tf.Tensor
        self.optimizer: tf.train.Optimizer
        self.loss: Tuple[tf.Tensor, tf.Tensor]
        self.accuracy: tf.Tensor
        self.global_step: tf.Tensor

    def compute_gradients(self, x, y):
        return self.sess.run([grad[0] for grad in self.grads],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 0.5})

    def apply_gradients(self, gradients: List[tf.Tensor]):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_loss_accuracy(self, x, y):
        return self.sess.run([self.loss, self.accuracy],
                             feed_dict={self.x: x,
                                        self.y_: y,
                                        self.keep_prob: 1.0})

    def add_helper_vars(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.variables = tf_variables.TensorFlowVariables(
            self.loss, self.sess)
        self.grads = self.optimizer.compute_gradients(
            self.loss)
        self.grads_placeholder = [
            (tf.placeholder("float", shape=grad[1].get_shape()), grad[1])
            for grad in self.grads]
        self.apply_grads_placeholder = self.optimizer.apply_gradients(
            self.grads_placeholder, global_step=self.global_step)
