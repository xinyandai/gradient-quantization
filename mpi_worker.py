from mpi4py import MPI
import mpi_model as model
import numpy as np


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
