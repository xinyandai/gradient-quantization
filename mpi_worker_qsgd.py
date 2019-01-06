from quantizer_scalar import ScalarCompressor
from mpi_worker import Worker
import numpy as np
import logging


class ScalarQuantizerWorker(Worker):
    def __init__(self, net, dataset, batch_size=64, test_size=1000, c_dim=1024, lr=1e-2):
        super(ScalarQuantizerWorker, self).__init__(net, dataset, batch_size, test_size, c_dim, lr)

        self.num_code = self.num_weights // c_dim // self.worker_size
        self.local_dim = self.num_code * c_dim
        self.compressed_dim = self.num_code * c_dim * self.worker_size
        self.uncompress_dim = self.num_weights - self.compressed_dim
        self.code_idx = np.array([i * self.num_code for i in range(self.worker_size + 1)])
        self.dim_idx = np.array([i * self.num_code * c_dim for i in range(self.worker_size + 1)])

        self.compressor = ScalarCompressor(
            size=self.compressed_dim,
            shape=(-1),
            c_dim=c_dim,
            s=256
        )

    def shuffle_reduce(self, gradients):
        """
        1/2. send compressed gradient shards to others
        1/2. receive compressed gradient shard from others
        3. decompressed and aggregate gradient shards
        4. send compressed reduced gradient shard back
        :param gradients:
        :return:
        """
        flat_grad = np.concatenate([g.flatten() for g in gradients])
        recv_grad = np.empty(shape=(self.worker_size, self.local_shard_size), dtype=np.float32)

        recv_norm = np.empty(shape=(self.worker_size, self.num_code), dtype=np.float32)
        recv_code = np.empty(shape=(self.worker_size, self.local_dim), dtype=np.uint8)
        recv_sign = np.empty(shape=(self.worker_size, self.local_dim // 8 + 1), dtype=np.uint8)

        [norms, signs, codes] = self.compressor.compress(flat_grad[:self.compressed_dim])
        [norms, signs, codes] = [norms, signs.reshape((-1, self.c_dim)), codes.reshape((-1, self.c_dim))]

        for i in range(self.worker_size):
            self.comm.Gather(norms[self.code_idx[i]:self.code_idx[i+1]], recv_norm, root=i)
            self.comm.Gather(codes[self.code_idx[i]:self.code_idx[i+1]], recv_code, root=i)

            pack_signs = np.packbits(signs[self.code_idx[i]:self.code_idx[i+1]])
            self.comm.Gather(pack_signs, recv_sign, root=i)

        for i in range(self.worker_size):
            if i != self.worker_index:
                norm = recv_norm[i, :]
                sign = np.unpackbits(recv_sign[i, :])[:self.local_dim]
                l = recv_code[i, :]
                recv_grad[i, :self.local_dim] = self.compressor.decompress([norm, sign, l]).flatten()
            else:
                recv_grad[i, :self.local_dim] = flat_grad[self.dim_idx[i]: self.dim_idx[i + 1]]

        # all reduce the reminders
        if self.worker_index == self.worker_size - 1:
            recv_others = np.empty(shape=(self.worker_size, self.uncompress_dim), dtype=np.float32)
        else:
            recv_others = None
        sendbuf = flat_grad[-self.uncompress_dim:]
        self.comm.Gather(sendbuf, recv_others, root=self.worker_size-1)

        if self.worker_index == self.worker_size - 1:
            recv_grad[:, self.local_dim:] = recv_others[:, :]

        self.apply_gradient(flat_grad, recv_grad)
