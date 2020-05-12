import math
import torch

from compressors import IdenticalCompressor

class RingQuantizer():
    def __init__(self, Compressor, parameters, args):
        self.parameters = list(parameters)
        self.num_layers = len(self.parameters)
        self.compressors = list()
        self.compressed_gradients = [list() for _ in range(self.num_layers)]
        self.args = args
        self.error_feedback = args.ef
        for param in self.parameters:
            param_size = param.flatten().shape[0]
            self.compressors.append(
                Compressor(param_size, param.shape, args) if param_size > 1000
                else IdenticalCompressor()
            )
            if self.error_feedback:
                param.error = [torch.zeros_like(param)
                               for _ in range(args.num_users)]

    def record(self, user, epoch):
        if self.args.scale == 'exp':
            scale = (2 / (math.exp(-epoch) + 1) - 1)
        else:
            scale = float(self.args.scale)
        for i, param in enumerate(self.parameters):
            if user != 0:
                param.grad.data.add_(self.compressed_gradients[i][-1])
            if self.error_feedback:
                param.grad.data.add_(scale * param.error[user])
                decompressed_g = self.compressors[i].decompress(
                    self.compressors[i].compress(param.grad.data)
                )
                param.error[user].data = param.grad.data - decompressed_g
            else:
                decompressed_g = self.compressors[i].decompress(
                    self.compressors[i].compress(param.grad.data)
                )
            self.compressed_gradients[i].append(decompressed_g)

    def apply(self):
        for i, param in enumerate(self.parameters):
            param.grad.data = self.compressed_gradients[i][-1]
        for compressed in self.compressed_gradients:
            compressed.clear()
