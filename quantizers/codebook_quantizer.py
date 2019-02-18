import numpy as np
import tensorflow as tf

from utils.vecs_io import fvecs_read
from utils.helper_func import normalize

from scipy.cluster.vq import vq
import logging

from .base_quantizer import BaseQuantizer


class CodebookQuantizer(BaseQuantizer):
    def __init__(self, placeholders):
        """
        :param placeholders: dict of variables
        """
        self.placeholders = placeholders
        self.layers = len(placeholders)
        self.pqs = [
            CodebookCompressor(
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
            for i, (pq, code) in enumerate(zip(self.pqs, gradient)):
                if aggregator[i] is None:
                    aggregator[i] = pq.decompress(code)
                else:
                    aggregator[i][:] += pq.decompress(code)
        for agg in aggregator:
            agg[:] = agg[:] / len(gradients)
        return aggregator


class CodebookCompressor(object):
    def __init__(self, size, shape, c_dim=16, ks=256):
        self.Ks = ks
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= c_dim else self.size
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (
            np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(
                self.dim, self.dim)
        _, self.codewords = normalize(fvecs_read(
            './codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks)))

    def compress(self, vec):
        vec = vec.reshape((-1, self.dim))
        norms, normalized_vecs = normalize(vec)
        codes, _ = vq(normalized_vecs, self.codewords)
        return [norms, codes.astype(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)
