from quantizer_codebook import CodebookCompressor
import numpy as np
import logging
from mpi_worker import Worker


class CodebookQuantizerWorker(Worker):
    def __init__(self, batch_size=64, test_size=1000, c_dim=16, lr=1e-2):
        super(CodebookQuantizerWorker, self).__init__(batch_size, test_size, c_dim, lr)

        self.num_code = self.num_weights // c_dim // self.worker_size
        self.local_dim = self.num_code * c_dim
        self.compressed_dim = self.num_code * c_dim * self.worker_size
        self.uncompress_dim = self.num_weights - self.compressed_dim
        self.code_idx = np.array([i * self.num_code for i in range(self.worker_size + 1)])
        self.dim_idx = np.array([i * self.num_code * c_dim for i in range(self.worker_size + 1)])

        logging.debug("gradients dimension  : {}".format(self.num_weights))
        logging.debug("uncompress dimension : {}".format(self.uncompress_dim))
        logging.debug("compressed dimension : {}".format(self.compressed_dim))
        logging.debug(" * num of code       : {}".format(self.num_code))
        logging.debug(" * size of dimension : {}".format(c_dim))
        logging.debug(" * num of workers    : {}".format(self.worker_size))

        self.compressor = CodebookCompressor(
            size=self.compressed_dim,
            shape=(-1),
            c_dim=c_dim
        )

    def shuffle_reduce(self, gradients):
        """
        * send compressed gradient shards(includes norms and codes) to others
        * receive compressed gradient shard from others
        * decompressed received norms and codes
        * aggregate gradient shards
        * For the uncompressed gradient:
        *   send uncompressed gradient to the last worker
        *   the last worker aggregate the received uncompressed gradients
        * send aggregated gradient shard back
        :param gradients:
        :return:
        """
        flat_grad = np.concatenate([g.flatten() for g in gradients])

        agg_grads = np.empty(shape=(self.worker_size, self.local_dim), dtype=np.float32)

        recv_norm = np.empty(shape=(self.worker_size, self.num_code), dtype=np.float32)
        recv_code = np.empty(shape=(self.worker_size, self.num_code), dtype=np.uint8)

        [norms, codes] = self.compressor.compress(flat_grad[:self.compressed_dim])

        for i in range(self.worker_size):
            self.comm.Gather(norms[self.code_idx[i]:self.code_idx[i+1]], recv_norm, root=i)
            self.comm.Gather(codes[self.code_idx[i]:self.code_idx[i+1]], recv_code, root=i)

        for i in range(self.worker_size):
            if i != self.worker_index:
                agg_grads[i, :] = self.compressor.decompress([recv_norm[i], recv_code[i]]).flatten()
            else:
                agg_grads[i, :] = flat_grad[self.dim_idx[i]: self.dim_idx[i + 1]]

        weights = self.net.variables.get_flat()
        weights[self.dim_idx[self.worker_index]: self.dim_idx[self.worker_index + 1]] \
            -= np.mean(agg_grads, axis=0) * self.lr

        # all reduce the reminders
        recv_others = np.empty(shape=(self.worker_size, self.uncompress_dim), dtype=np.float32)
        sendbuf = flat_grad[-self.uncompress_dim:]
        self.comm.Gather(sendbuf, recv_others, root=self.worker_size-1)
        if self.worker_index == self.worker_size - 1:
            weights[-self.uncompress_dim:] = np.mean(recv_others, axis=0) * self.lr

        self.syn_weights(weights)
