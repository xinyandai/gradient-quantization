import torch
import numpy as np
from scipy import stats

from utils.vecs_io import fvecs_read
from utils.vec_np import normalize
from .probabilistic_scalar_compressor import ProbabilisticScalarCompressor

class NearestNeighborCompressor(object):
    def __init__(self, size, shape, args):
        c_dim = args.c_dim
        k_bit = args.k_bit
        n_bit = args.n_bit
        compressed_norm = n_bit != 32
        assert c_dim > 0
        assert k_bit >= 0
        assert n_bit > 0

        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape

        if c_dim == 0 or self.size < args.c_dim:
            self.dim = self.size
        else:
            self.dim = c_dim
            for i in range(0, 10):
                if size % self.dim != 0:
                    self.dim = self.dim // 2 * 3

        if c_dim != self.dim:
            print("alternate dimension form"
                  " {} to {}, size {} shape {}"
                  .format(c_dim, self.dim, size, shape))

        assert size % self.dim == 0, \
            "not divisible size {}  c_dim {} self.dim {}"\
                .format(size, c_dim, self.dim)

        if k_bit <= 0:
            self.K = self.dim
        else:
            self.K = 2 ** k_bit
        # self.codewords = normalize(np.random.normal(size=(self.K, self.dim)))[1].astype(np.float32)
        if self.K == self.dim:
            self.codewords = stats.ortho_group.rvs(self.dim).astype(np.float32)
            # self.codewords = np.identity(self.dim, dtype=np.float32)
            # self.codewords = normalize(np.random.normal(size=(self.dim, self.dim)))[1].astype(np.float32)
        else:
            location = './codebooks/learned_codebook/' \
                       'angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.K)
            _, self.codewords = normalize(fvecs_read(location))

        self.codewords = torch.from_numpy(self.codewords)
        if self.cuda:
            self.codewords = self.codewords.cuda()
        self.code_dtype = torch.uint8 if k_bit <= 8 else torch.int32

        self.compressed_norm = compressed_norm
        if self.compressed_norm:
            self.norm_compressor = ProbabilisticScalarCompressor(n_bit, args)

    def compress(self, vec):

        vec = vec.view(-1, self.dim)

        # calculate probability, complexity: O(d*K)
        p = torch.mm(self.codewords, vec.transpose(0, 1)).transpose(0, 1)
        probability = torch.abs(p)

        # choose codeword
        codes = torch.argmax(probability, dim=1)
        u = p.gather(dim=1, index=codes.view(-1, 1)).view(-1)

        if self.compressed_norm:
            u = self.norm_compressor.compress(u)

        return [u, codes.type(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        if self.compressed_norm:
            norms =  self.norm_compressor.decompress(norms)

        codes = codes.view(-1).type(torch.long)
        norms = norms.view(-1)

        vec = self.codewords[codes]
        recover = torch.mul(vec, norms.view(-1, 1).expand_as(vec))
        return recover.view(self.shape)

