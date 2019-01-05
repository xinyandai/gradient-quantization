from quantizer_scalar import ScalarCompressor
from mpi_worker import Worker
import numpy as np
import logging


class ScalarQuantizerWorker(Worker):
    def __init__(self, net, dataset, batch_size=64, test_size=1000, c_dim=16, lr=1e-2):
        super(ScalarQuantizerWorker, self).__init__(net, dataset, batch_size, test_size, c_dim, lr)
        self.compressor = ScalarCompressor()

    @staticmethod
    def compressed_size(float_size):
        """
        :param float_size: number of float integers before compressing
        :return: number of uint8 afters compressing
        """
        return float_size + float_size // 8 + 1 + 4

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
        recv_buff = np.empty(shape=(self.worker_size, self.compressed_size(self.local_shard_size)), dtype=np.uint8)

        for i in range(self.worker_size):
            shard_i = flat_grad[self.shards[i]: self.shards[i + 1]]
            [norm, signs, l] = self.compressor.compress(vec=shard_i)

            sendbuf = np.empty(shape=self.compressed_size(len(shard_i)), dtype=np.uint8)
            sendbuf[0:4].view(np.float32)[0] = norm
            sendbuf[4:4+len(shard_i)] = l[:]
            pack_signs = np.packbits(signs)
            sendbuf[4+len(shard_i):4+len(shard_i) + len(pack_signs)] = pack_signs[:]

            self.comm.Gather(sendbuf, recv_buff, root=i)

        for i in range(self.worker_size):
            if i != self.worker_index:
                norm = recv_buff[i, 0:4].view(np.float32)[0]
                signs = np.unpackbits(recv_buff[i, 4 + self.local_shard_size:])[:self.local_shard_size]
                l = recv_buff[i, 4:4 + self.local_shard_size]

                recv_grad[i, :] = self.compressor.decompress(norm=norm, signs=signs, l=l)
            else:
                recv_grad[i, :] = flat_grad[self.shards[i]: self.shards[i + 1]]

        self.apply_gradient(flat_grad, recv_grad)
