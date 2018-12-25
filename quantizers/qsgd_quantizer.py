from .base_quantizer import BaseQuantizer
import torch
import numpy as np
from compressors import QSGDCompressor


class QSGDQuantizer(BaseQuantizer):
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        for param in self.parameters:
            self.compressors.append(
                QSGDCompressor(
                    param.flatten().shape[0],
                    param.shape
                )
            )

    def record(self):
        for i, param in enumerate(self.parameters):
            self.compressed_gradients[i].append(
                self.compressors[i].compress(param.grad.data.cpu().numpy()))

    def apply(self):
        for i, param in enumerate(self.parameters):
            decompressed_gradients = [self.compressors[i].decompress(
                compressed) for compressed in self.compressed_gradients[i]]
            param.grad.data = torch.from_numpy(
                np.stack(decompressed_gradients, axis=0).mean(axis=0)).cuda()
        for compressed in self.compressed_gradients:
            compressed.clear()



