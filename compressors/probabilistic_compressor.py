import torch


class ProbabilisticCompressor(object):
    def __init__(self, n_bit, args):
        self.s = 2 ** n_bit
        self.cuda = not args.no_cuda
        self.code_dtype = torch.uint8 if n_bit <= 8 else torch.int32
        self.random = args.random


    def compress(self, vec):
        lower_bound = torch.min(vec)
        upper_bound = torch.max(vec)
        if lower_bound - upper_bound == 0.0:
            return lower_bound, upper_bound, torch.zeros_like(vec).astype(torch.int32)
        scaled_vec = torch.abs((vec - lower_bound) / (upper_bound - lower_bound)) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(dtype=self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.cuda()
            l[:] += probabilities > r
        return lower_bound, upper_bound, l

    def decompress(self, signature):
        [lower_bound, upper_bound, l] = signature
        scaled_vec = l.type(dtype=torch.float32)
        compressed = scaled_vec / self.s * (upper_bound - lower_bound) + lower_bound
        return compressed
