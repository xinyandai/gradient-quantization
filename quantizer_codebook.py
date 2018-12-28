import numpy as np
import tensorflow as tf
from vecs_io import fvecs_read
from scipy.cluster.vq import vq


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)
    # divide by zero problem
    return norms, normalized_vecs


class ProductNorms(object):
    def __init__(self, size, shape):
        self.Ks = 256
        self.size = size
        self.shape = shape
        self.dim = 16 if self.size >= 16 else self.size
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)
        self.codewords = fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks))

    def encode(self, vec):

        codes = np.empty(self.M, dtype=self.code_dtype)
        vec = vec.reshape((self.M, self.dim))
        norms, normalized_vecs = normalize(vec)

        codes[:], _ = vq(normalized_vecs, self.codewords)
        return [norms, codes]

    def decode(self, signature):
        [norms, codes] = signature
        vec = np.empty((self.M, self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)


class ProductNormsReminder(object):
    def __init__(self, size, shape):
        self.Ks = 256
        self.dim = 16
        self.size = size
        self.shape = shape
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.reminder = size % self.dim
        self.M = size // self.dim

        if self.reminder == 0:
            self.reminder = self.dim
        else:
            self.M += 1

        self.codewords = fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks))
        self.codewords_reminder = fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.reminder, self.Ks))

        self.Ds = [i * self.dim for i in range(self.M+1)]
        self.Ds[-1] = self.size

    def encode(self, vec):

        codes = np.empty(self.M, dtype=self.code_dtype)
        norms = np.empty(self.M, dtype=np.float32)
        vec = vec.reshape((-1))

        for m in range(self.M):
            vecs_sub = vec[self.Ds[m]: self.Ds[m + 1]]
            norms[m] = np.linalg.norm(vecs_sub)
            if norms[m] != 0.0:
                vecs_sub[:] = vecs_sub[:] / norms[m]

        codes[:-1], _ = vq(
            vec[:-self.reminder].reshape((-1, self.dim)),
            self.codewords
        )
        codes[-1:], _ = vq(
            vec[-self.reminder:].reshape((1, self.reminder)),
            self.codewords_reminder
        )

        return [norms, codes]

    def decode(self, signature):
        [norms, codes] = signature
        vec = np.empty(self.size, dtype=np.float32)
        for m in range(0, self.M-1):
            vec[self.Ds[m]: self.Ds[m + 1]] = norms[m] * self.codewords[codes[m]]
        vec[-self.reminder:] = norms[-1] * self.codewords_reminder[codes[-1]]

        return vec.reshape(self.shape)


Compressor = ProductNorms


class CodebookQuantizer(object):
    def __init__(self, placeholders):
        """
        :param placeholders: dict of variables
        """
        self.placeholders = placeholders
        self.layers = len(placeholders)
        self.pqs = [
            Compressor(
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
