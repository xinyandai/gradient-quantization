import numpy as np


class IdenticalQuantizer(object):
    def __init__(self, placeholders):
        self.placeholders = placeholders

    def encode(self, gradient):
        assert len(self.placeholders) == len(gradient)
        return gradient

    def decode(self, gradients):
        for gradient in gradients:
            assert len(self.placeholders) == len(gradient)
        return np.mean(gradients, axis=0)
