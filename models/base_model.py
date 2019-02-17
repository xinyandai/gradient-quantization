import tensorflow as tf


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


# Test
if __name__ == "__main__":
    model = BaseModel()
    print("No error raised.")
