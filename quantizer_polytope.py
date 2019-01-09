import numpy as np
import tensorflow as tf


class PolytopeCompressor(object):
    def __init__(self, size, shape, c_dim=128, ks=128):
        self.Ks = ks
        self.size = size
        self.shape = shape
        self.dim = c_dim
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 7 else (np.uint16 if self.Ks <= 2 ** 15 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)

        self.codewords = np.concatenate((np.eye(self.dim), -np.eye(self.dim)))

    def compress(self, vec):
        vec = vec.reshape((-1, self.dim))
        norms = np.linalg.norm(vec, axis=1)
        codes = np.argmax(np.abs(vec), axis=1)
        sign_flip = (1 - np.sign(vec[np.arange(len(vec)), codes]).astype(dtype=np.int32)) * self.dim // 2
        assert np.all(sign_flip >= 0)
        codes += sign_flip
        return [norms, codes.astype(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)


class PolytopeQuantizer(object):
    def __init__(self, placeholders):
        """
        :param placeholders: dict of variables
        """
        self.placeholders = placeholders
        self.layers = len(placeholders)
        self.pqs = [
            PolytopeCompressor(
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
