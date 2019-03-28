import torch


class QSGDCompressor(object):
    def __init__(self, size, shape, c_dim=256, s=256, random=True):
        self.random = random
        self.s = s
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= (4096*2) else self.size
        self.M = size // self.dim
        self.code_dtype = torch.uint8 if self.s <= 2 ** 8 else torch.int32
        assert size % self.dim == 0, \
            "dimension of variable {} {} should be smaller than {} or dividable by {}".format(
                shape, size, self.dim, self.dim)

    def compress(self, vec):
        """
        :param vec: torch tensor
        :return: norm, signs, quantized_intervals
        """
        vec = vec.view(-1, self.dim)
        norm = torch.norm(vec, dim=1, keepdim=True)
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            l[:] += probabilities > torch.rand(l.size()).cuda()

        signs = torch.sign(vec) > 0
        return [norm, signs.view(self.shape), l.view(self.shape)]

    def decompress(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        scaled_vec = l.type(torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1, self.dim))) * norm / self.s
        return compressed.view(self.shape)
