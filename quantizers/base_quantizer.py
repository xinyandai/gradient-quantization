import torch
from compressors import IdenticalCompressor

class Quantizer():
    def __init__(self, Compressor, parameters, args):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        self.args = args
        for param in self.parameters:
            param_size = param.flatten().shape[0]
            self.compressors.append(
                Compressor(param_size, param.shape, args) if param_size > 1000
                else IdenticalCompressor()
            )

    def record(self):
        for i, param in enumerate(self.parameters):
            self.compressed_gradients[i].append(
                self.compressors[i].compress(param.grad.data))

    def apply(self):
        for i, param in enumerate(self.parameters):
            decompressed_gradients = [self.compressors[i].decompress(
                compressed) for compressed in self.compressed_gradients[i]]
            g = torch.stack(decompressed_gradients, dim=0).mean(dim=0)

            # if compress gradient on two phase, i.e., compress the sum of decompressed gradient
            if self.args.two_phase:
                g = self.compressors[i].decompress(
                        self.compressors[i].compress(g))
            param.grad.data = g
        for compressed in self.compressed_gradients:
            compressed.clear()
