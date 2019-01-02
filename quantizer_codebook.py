import numpy as np
import tensorflow as tf
from vecs_io import fvecs_read
from scipy.cluster.vq import vq
import logging


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)
    # divide by zero problem
    return norms, normalized_vecs


class CodebookCompressor(object):
    def __init__(self, size, shape, c_dim=8):
        self.Ks = 256
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= 16 else self.size
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)
        _, self.codewords = normalize(fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks)))

    def compress(self, vec):
        vec = vec.reshape((-1, self.dim))
        norms, normalized_vecs = normalize(vec)
        codes, _ = vq(normalized_vecs, self.codewords)
        return [norms, codes.astype(np.uint8)]

    def decompress(self, signature):
        [norms, codes] = signature
        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec

    def encode(self, vec):
        return self.compress(vec)

    def decode(self, signature):
        return self.decompress(signature).reshape(self.shape)


class CodebookQuantizer(object):
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
        return [pq.encode(g) for pq, g in zip(self.pqs, gradient)]

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
                    aggregator[i] = pq.decode(code)
                else:
                    aggregator[i][:] += pq.decode(code)
        for agg in aggregator:
            agg[:] = agg[:] / len(gradients)
        return aggregator
