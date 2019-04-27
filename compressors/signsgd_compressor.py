import torch


class SignSGDCompressor(object):
    def __init__(self, size, shape, args):
        pass

    def compress(self, vec):
        return torch.sign(vec)

    def decompress(self, signature):
        return signature
