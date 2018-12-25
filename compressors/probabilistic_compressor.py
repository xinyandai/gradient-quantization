import numpy as np


class ProbabilisticCompressor(object):
    def __init__(self, s):
        self.s = s
        self.random = True

    def compress(self, vec):
        lower_bound = np.min(vec)
        upper_bound = np.max(vec)
        if lower_bound - upper_bound == 0.0:
            return lower_bound, upper_bound, np.zeros_like(vec).astype(np.int32)
        scaled_vec = np.abs((vec - lower_bound) / (upper_bound - lower_bound)) * self.s
        l = np.array(scaled_vec).clip(0, self.s-1).astype(dtype=np.int32)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l
            l[:] += probabilities > np.random.uniform(0, 1, l.shape)
        return lower_bound, upper_bound, l

    def decompress(self, signature):
        [lower_bound, upper_bound, l] = signature
        scaled_vec = l.astype(dtype=np.float32)
        compressed = scaled_vec / self.s * (upper_bound - lower_bound) + lower_bound
        return compressed
