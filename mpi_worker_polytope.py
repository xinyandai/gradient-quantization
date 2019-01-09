from quantizer_polytope import PolytopeCompressor
from mpi_worker_vqsgd import CodebookQuantizerWorker


class PolytopeQuantizerWorker(CodebookQuantizerWorker):
    def __init__(self, net, dataset, batch_size=64, test_size=1000, c_dim=64, lr=1e-2):
        super(PolytopeQuantizerWorker, self).__init__(net, dataset, batch_size, test_size, c_dim, lr)

        self.compressor = PolytopeCompressor(
            size=self.compressed_dim,
            shape=(-1),
            c_dim=c_dim
        )
