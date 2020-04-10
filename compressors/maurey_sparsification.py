import torch


class MaureySparsification(object):
    def __init__(self, size, shape, args):

        self.cr = 32 * args.c_dim // (args.k_bit + args.n_bit)
        bit_for_idx = 32 if size > 65536 else 16
        self.k = 32 * size // ((bit_for_idx + 1) * self.cr)
        # self.cr = 32 * size / ((bit_for_idx + 1) *self.k)
        if self.k == 0:
            self.k = 1
        assert self.k > 0
        print("MaureySparsification size: {} c_dim {} k-bit: {} n-bit: {} cr: {} K: {}".format(
            size, args.c_dim, args.k_bit, args.n_bit, self.cr, self.k))
        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.idx = torch.range(1, size)

    def compress(self, vec):
        vec = vec.view(-1)
        l1_norm = torch.norm(vec, p=1)
        # calculate probability, complexity: O(d*K)
        probability = torch.abs(vec) / l1_norm
        d = probability.size(0)
        # choose codeword with probability (on cpu)
        r = torch.rand((self.k, d))
        comp = torch.cumsum(probability, dim=0).cpu() >= r.expand(size=(self.k, d))
        comp = comp.type(torch.float32) / self.idx # shape = (k by d)

        codes = torch.argmax(comp, dim=1) # shape = (k)

        selected_p = vec.cpu().gather(index=codes, dim=0) # shape = (k)
        signs = torch.sign(selected_p)
        return [l1_norm / self.k, codes, signs]

    def decompress(self, signature):
        [scale, codes, signs] = signature

        sparse = torch.sparse_coo_tensor(values=signs,
                                         size=(self.k, self.size),
                                         indices= torch.stack([
                                            torch.arange(0, self.k),
                                            codes.view(self.k).type(torch.long)]),)
        recover = sparse.to_dense().sum(dim=0)
        recover = scale * recover.view(self.shape)
        if self.cuda:
            recover = recover.cuda()
        return recover
