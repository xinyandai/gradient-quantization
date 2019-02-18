import numpy as np
import tensorflow as tf

from myutils import normalize
from .base_quantizer import BaseQuantizer


class ScalarQuantizer(BaseQuantizer):
    def __init__(self, placeholders):
        """
        :param placeholders: dict of variables
        {'conv1/Variable': <tf.Tensor 'Placeholder_conv1/Variable:0' shape=(5, 5, 1, 32) dtype=float32>,
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

        self.pqs = [
            ScalarCompressor(
                tf.reshape(v, [-1]).shape.as_list()[0],
                v.shape.as_list()
            )
            for _, v in placeholders.items()
        ]

    def encode(self, gradient):
        """
        :param gradient: python list of numpy arrays
        :return: python list of python list
        """
        assert self.layers == len(gradient)
        return [pq.compress(g) for pq, g in zip(self.pqs, gradient)]

    def decode(self, gradients):
        """
        :param gradients:
        :return:
        """
        for gradient in gradients:
            assert self.layers == len(gradient)

        aggregator = [None for _ in range(self.layers)]

        for gradient in gradients:
            for i, [norm, signs, l] in enumerate(gradient):
                if aggregator[i] is None:
                    aggregator[i] = self.pqs[i].decompress([norm, signs, l])
                else:
                    aggregator[i][:] += self.pqs[i].decompress(
                        [norm, signs, l])

        for agg in aggregator:
            agg[:] = agg[:] / len(gradients)
        return aggregator


class ScalarCompressor(object):
    def __init__(self, size, shape, c_dim=1024, s=256, random=True):
        self.random = random
        self.s = s
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= c_dim else self.size
        self.M = size // self.dim
        self.code_dtype = np.uint8 if self.s <= 2 ** 8 else (
            np.uint16 if self.s <= 2 ** 16 else np.uint32)
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(
                self.dim, self.dim)

    def compress(self, vec):
        """
        :param vec: numpy array
        :return: norm, signs, quantized_intervals
        """
        norm, normalized_vec = normalize(vec.reshape((-1, self.dim)))
        scaled_vec = np.abs(normalized_vec) * self.s
        l = np.array(scaled_vec).clip(
            0, self.s-1).astype(dtype=self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l
            l[:] += probabilities > np.random.uniform(0, 1, l.shape)

        signs = np.sign(vec) > 0

        return [norm, signs.reshape(self.shape), l.reshape(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.astype(dtype=np.float32) * \
            (2 * signs.astype(dtype=np.float32) - 1)
        compressed = ((scaled_vec.reshape((-1, self.dim))).T * norm / self.s).T
        return compressed.reshape(self.shape)
