import torch

from .probabilistic_vector_compressor import ProbabilisticVectorCompressor
from .nearest_neighbor_compressor import NearestNeighborCompressor


class ResidualCompressor(object):
    def __init__(self, size, shape, args):
        self.compressors = [
            NearestNeighborCompressor(size, shape, args),
            ProbabilisticVectorCompressor(size, shape, args),
        ]


    def compress(self, vec):
        residuals = vec.clone()
        signatures = []
        for compressor in self.compressors:
            signature = compressor.compress(residuals)
            decompressed = compressor.decompress(signature)
            residuals -= decompressed
            signatures.append(signature)

        return signatures

    def decompress(self, signatures):
        decompressed_gradients = [
            compressor.decompress(signature)
            for signature, compressor
            in zip(signatures, self.compressors)
        ]
        return torch.stack(decompressed_gradients, dim=0).sum(dim=0)

