import torch
from compressors.identical_compressor import IdenticalCompressor
from .base_quantizer import BaseQuantizer


class SGDQuantizer(BaseQuantizer):
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        for _ in self.parameters:
            self.compressors.append(
                IdenticalCompressor()
            )

    def record(self):
        for i, param in enumerate(self.parameters):
            self.compressed_gradients[i].append(
                self.compressors[i].compress(param.grad))

    def apply(self):
        for i, param in enumerate(self.parameters):
            decompressed_gradients = [self.compressors[i].decompress(
                compressed) for compressed in self.compressed_gradients[i]]
            param.grad = torch.stack(decompressed_gradients, dim=0).mean(dim=0)
        for compressed in self.compressed_gradients:
            compressed.clear()

