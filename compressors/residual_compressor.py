import torch

from .hyper_sphere_compressor import HyperSphereCompressor
from .nearest_neighbor_compressor import NearestNeighborCompressor


class ResidualCompressor(object):
    def __init__(self, size, shape, c_dim=64, k=256, compressed_norm=True):
        self.size = size
        self.shape = shape
        self.dim = c_dim if c_dim < size else size
        self.K = k
        self.compressors = [
            NearestNeighborCompressor(
                size=size, shape=shape, c_dim=c_dim, k=k, compressed_norm=compressed_norm),
            HyperSphereCompressor(
                size=size, shape=shape, c_dim=c_dim, k=k, compressed_norm=compressed_norm),
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

