import torch
from compressors import IdenticalCompressor

class Quantizer():
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

    def record(self, user):
        for i, param in enumerate(self.parameters):
            if self.error_feedback:
                param.grad.data.add_(param.error[user])
            decompressed_g = self.compressors[i].decompress(
                self.compressors[i].compress(param.grad.data)
            )
            self.compressed_gradients[i].append(decompressed_g)
            if self.error_feedback:
                param.error[user].data = param.grad.data - decompressed_g

    def apply(self):
        for i, param in enumerate(self.parameters):
            g = torch.stack(self.compressed_gradients[i], dim=0).mean(dim=0)

            # if compress gradient on two phase, i.e., compress the sum of decompressed gradient
            if self.args.two_phase:
                g = self.compressors[i].decompress(
                        self.compressors[i].compress(g))
            param.grad.data = g
        for compressed in self.compressed_gradients:
            compressed.clear()
