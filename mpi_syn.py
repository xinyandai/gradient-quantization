import mpi_model as model
from mpi4py import MPI
from quantizer_scalar import ScalarCompressor
import numpy as np
import logging


class Worker(object):
    def __init__(self, batch_size=64, test_size=1000, c_dim=16):
        self.comm = MPI.COMM_WORLD
        self.worker_index = self.comm.Get_rank()
        self.worker_size = self.comm.Get_size()

        self.batch_size = batch_size
        self.test_size = test_size

        self.mnist = model.download_mnist_retry(seed=self.worker_index)
        self.net = model.SimpleCNN()

        self.num_weights = self.net.variables.get_flat_size()

        shard_size = self.num_weights // c_dim // self.worker_size
        self.shards = np.array([i * shard_size * c_dim for i in range(self.worker_size + 1)])
        self.shards[-1] = self.num_weights
        self.local_shard_size = self.shards[self.worker_index + 1] - self.shards[self.worker_index]

        self.tag_weights = 808
        self.tag_gradients = 168

    def syn_weights(self, weights):
        """
        1. copy local aggregated shard weights to remote
        2. update other shard weights from remote
        3. update variables
        :return:
        """
        for i in range(self.worker_size):
            received = self.comm.bcast(weights[self.shards[i]: self.shards[i+1]], root=i)
            if i != self.worker_index:
                weights[self.shards[i]: self.shards[i + 1]] = received[:]
        self.net.variables.set_flat(weights)

    def compute_gradients(self):
        """
        :return: a python list of numpy array
        """
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)

    def shuffle_reduce(self, gradients):
        """
        1/2. send gradient shards to others
        1/2. receive gradient shard from others
        3. aggregate gradient shards
        4. send reduced gradient shard back
        :param gradients:
        :return:
        """
        flat_grad = np.concatenate([g.flatten() for g in gradients])
        recvbuf = np.empty(shape=(self.worker_size, self.local_shard_size), dtype=np.float32)
        for i in range(self.worker_size):
            sendbuf = flat_grad[self.shards[i]: self.shards[i + 1]]
            self.comm.Gather(sendbuf, recvbuf, root=i)

        weights = self.net.variables.get_flat()
        weights[self.shards[self.worker_index]: self.shards[self.worker_index + 1]] -= np.mean(recvbuf, axis=0) * 1e-2

        self.syn_weights(weights)

    def compute_loss_accuracy(self):
        xs, ys = self.mnist.train.next_batch(self.test_size)
        return self.net.compute_loss_accuracy(xs, ys)


class ScalarQuantizerWorker(Worker):
    def __init__(self, batch_size=64, test_size=1000, c_dim=16):
        super(ScalarQuantizerWorker, self).__init__(batch_size, test_size, c_dim)
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

            if self.worker_index == 0 and i == 1:
                logging.debug("compress \n[{}]\n "
                              "as \n[{}]\n  "
                              "norm \n[{}]\n"
                              "signs \n[{}]\n"
                              "unpack signs \n[{}]\n"
                              "quantized \n[{}]\n"
                              "from rank[{}] to [{}]".format(
                    shard_i[:10],
                    self.compressor.decompress(norm, signs, l)[:10],
                    norm,
                    signs[:10],
                    np.unpackbits(pack_signs)[:10],
                    l[:10],
                    0,
                    1))

            self.comm.Gather(sendbuf, recv_buff, root=i)

        for i in range(self.worker_size):
            if i != self.worker_index:
                norm = recv_buff[i, 0:4].view(np.float32)[0]
                signs = np.unpackbits(recv_buff[i, 4 + self.local_shard_size:])[:self.local_shard_size]
                l = recv_buff[i, 4:4 + self.local_shard_size]

                recv_grad[i, :] = self.compressor.decompress(norm=norm, signs=signs, l=l)
                if self.worker_index == 1 and i == 0:

                    logging.debug("as \n[{}]\n  "
                                  "norm \n[{}]\n"
                                  "unpack signs \n[{}]\n"
                                  "quantized \n[{}]\n"
                                  "from rank[{}] to [{}]".format(
                        self.compressor.decompress(norm, signs, l)[:10],
                        norm,
                        signs[:10],
                        l[:10],
                        0,
                        1))
            else:
                recv_grad[i, :] = flat_grad[self.shards[i]: self.shards[i + 1]]

        weights = self.net.variables.get_flat()
        weights[self.shards[self.worker_index]: self.shards[self.worker_index + 1]] -= np.mean(recv_grad, axis=0) * 1e-2

        self.syn_weights(weights)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.debug('initialize worker')
    worker = ScalarQuantizerWorker()
    logging.debug('synchronize weights before running.')
    worker.syn_weights(worker.net.variables.get_flat())
    i = 0
    while i <= 50:
        if i % 10 == 0:
            # Evaluate the current model.
            loss, accuracy = worker.compute_loss_accuracy()
            logging.warning("Iteration {}: loss is {}, accuracy is {}".format(i, loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
