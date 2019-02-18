
import numpy as np
from .base_quantizer import BaseQuantizer


class IdenticalQuantizer(BaseQuantizer):
    def __init__(self, placeholders):
        super(IdenticalQuantizer).__init__()
        self.placeholders = placeholders

    def encode(self, gradient):
        assert len(self.placeholders) == len(gradient)
        return gradient

    def decode(self, gradients):
        for gradient in gradients:
            assert len(self.placeholders) == len(gradient)
        return np.mean(gradients, axis=0)
