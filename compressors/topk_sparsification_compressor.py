import torch
import numpy as np
from scipy import stats

from utils.vecs_io import fvecs_read
from utils.vec_np import normalize
from .probabilistic_scalar_compressor import ProbabilisticScalarCompressor

class TopKSparsificationCompressor(object):
    def __init__(self, size, shape, args):
        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.users = 1
        # self.k = size // 146 // 2
        self.k = size // args.cr 

    def compress(self, vec):
        vec = vec.view(self.users, -1)
        ind = torch.zeros_like(vec)
        idx = torch.topk(torch.abs(vec), k=self.k, dim=1)[1]
        ind.scatter_(1, idx, 1)
        return vec * ind

    def decompress(self, signature):
        return signature.view(self.shape)
