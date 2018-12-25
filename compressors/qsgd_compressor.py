import numpy as np
from utils.vec_np import normalize


class QSGDCompressor(object):
    def __init__(self, size, shape, c_dim=512, s=256, random=True):
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
