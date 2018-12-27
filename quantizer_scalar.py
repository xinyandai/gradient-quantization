import numpy as np


class ScalarQuantizer(object):
    def __init__(self, placeholders):
        """
        :param placeholders: dict of variables
         eg. {'conv1/Variable': <tf.Tensor 'Placeholder_conv1/Variable:0' shape=(5, 5, 1, 32) dtype=float32>,
             'conv1/Variable_1': <tf.Tensor 'Placeholder_conv1/Variable_1:0' shape=(32,) dtype=float32>,
             'conv2/Variable': <tf.Tensor 'Placeholder_conv2/Variable:0' shape=(5, 5, 32, 64) dtype=float32>,
             'conv2/Variable_1': <tf.Tensor 'Placeholder_conv2/Variable_1:0' shape=(64,) dtype=float32>,
             'fc1/Variable': <tf.Tensor 'Placeholder_fc1/Variable:0' shape=(3136, 1024) dtype=float32>,
             'fc1/Variable_1': <tf.Tensor 'Placeholder_fc1/Variable_1:0' shape=(1024,) dtype=float32>,
             'fc2/Variable': <tf.Tensor 'Placeholder_fc2/Variable:0' shape=(1024, 10) dtype=float32>,
             'fc2/Variable_1': <tf.Tensor 'Placeholder_fc2/Variable_1:0' shape=(10,) dtype=float32>}
        """
        self.placeholders = placeholders
        self.layers = len(placeholders)
        self.s = 511

    def _encode(self, vec):
        """
        :param vec: numpy array
        :return:
        """
        norm = np.linalg.norm(vec)
        quantized = np.array(vec * self.s).astype(dtype=np.int8)
        return [norm, quantized]

    def encode(self, gradient):
        """
        :param gradient: python list of numpy arrays
        :return: python list of python list
        """
        assert self.layers == len(gradient)
        return [self._encode(g) for g in gradient]

    def decode(self, gradients):
        """
        :param gradients:
        :return:
        """
        for gradient in gradients:
            assert self.layers == len(gradient)

        aggregator = [None for _ in range(self.layers)]

        for gradient in gradients:
            for i, [norm, quantized] in enumerate(gradient):
                if aggregator[i] is None:
                    aggregator[i] = quantized.astype(dtype=np.float32) * norm
                else:
                    aggregator[i][:] += quantized.astype(dtype=np.float32) * norm
        for agg in aggregator:
            agg[:] = agg[:] / len(gradients)
        return aggregator