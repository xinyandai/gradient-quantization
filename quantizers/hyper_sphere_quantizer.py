import torch
# from .base_quantizer import BaseQuantizer
from compressors.hyper_sphere_compressor import HyperSphereCompressor


class HyperSphereQuantizer(object):
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        for param in self.parameters:
            self.compressors.append(
                HyperSphereCompressor(
                    param.flatten().shape[0],
                    param.shape
                )
            )

    def record(self):
        for i, param in enumerate(self.parameters):
            self.compressed_gradients[i].append(
                self.compressors[i].compress(param.grad.data))

    def apply(self):
        for i, param in enumerate(self.parameters):
            decompressed_gradients = [self.compressors[i].decompress(
                compressed) for compressed in self.compressed_gradients[i]]
            param.grad.data = torch.stack(decompressed_gradients, dim=0).mean(dim=0)
        for compressed in self.compressed_gradients:
            compressed.clear()

