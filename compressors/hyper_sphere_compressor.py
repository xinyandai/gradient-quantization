import torch
import numpy as np
from scipy import stats
from utils.vecs_io import fvecs_read
from utils.vec_np import normalize
from .probabilistic_compressor import ProbabilisticCompressor

class HyperSphereCompressor(object):
    def __init__(self, size, shape, args):
        c_dim = args.c_dim
        k_bit = args.k_bit
        n_bit = args.n_bit
        compressed_norm = n_bit != 32
        assert c_dim > 0
        assert k_bit > 0
        assert n_bit > 0

        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.dim = c_dim if c_dim < size else size
        self.K = 2 ** k_bit
        if self.K == self.dim:
            self.codewords = stats.ortho_group.rvs(self.dim).astype(np.float32)
        else:
            location = './codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.K)
            _, self.codewords = normalize(fvecs_read(location))
        self.c_dagger = np.linalg.pinv(self.codewords.T)

        self.codewords = torch.from_numpy(self.codewords)
        self.c_dagger = torch.from_numpy(self.c_dagger)

        if self.cuda:
            self.codewords = self.codewords.cuda()
            self.c_dagger = self.c_dagger.cuda()

        self.code_dtype = torch.uint8 if k_bit <=  8 else torch.int32
        self.compressed_norm = compressed_norm
        if self.compressed_norm:
            self.norm_compressor = ProbabilisticCompressor(n_bit, args)

    def compress(self, vec):

        vec = vec.view(-1, self.dim)

        # calculate probability, complexity: O(d*K)
        p = torch.mm(self.c_dagger, vec.transpose(0, 1)).transpose(0, 1)
        l1_norms = torch.norm(p, p=1, dim=1, keepdim=True)
        probability = torch.abs(p) / l1_norms

        # choose codeword with probability
        r = torch.rand(probability.size()[0])
        if self.cuda:
            r = r.cuda()
        rs = r.view(-1, 1).expand_as(probability)

        comp = torch.cumsum(probability, dim=1) >= rs-(1e-5)
        codes = torch.argmin(comp, dim=1) + 1

        selected_p = p.gather(dim=1, index=codes.view(-1, 1))
        u = torch.mul(torch.sign(selected_p.view(-1)), l1_norms.view(-1))

        if self.compressed_norm:
            u = self.norm_compressor.compress(u)
        return [u, codes.type(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        if self.compressed_norm:
            norms = self.norm_compressor.decompress(norms)

        codes = codes.view(-1).type(torch.long)
        norms = norms.view(-1)

        vec = self.codewords[codes]
        recover = torch.mul(vec, norms.view(-1, 1).expand_as(vec))
        return recover.view(self.shape)
