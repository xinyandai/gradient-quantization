from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import argparse
import mpi_dataset

import numpy as np

from myutils import Timer

from model_resnet import ResNet
from model_simple import SimpleCNN
from model_lr import LinearRegression

from quantizer_identical import IdenticalQuantizer
from quantizer_scalar import ScalarQuantizer
from quantizer_codebook import CodebookQuantizer


Quantizer = None
parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=1, type=int,
                    help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--quantizer", default='codebook', type=str,
                    help="Compressor for gradient.")
parser.add_argument("--two-phases", default=False, type=bool,
                    help="Using 2-phases quantization.")
parser.add_argument("--network", default="resnet", type=str,
                    help="Network architectures")
parser.add_argument("--batch-size", default=128, type=int,
                    help="batch size.")
parser.add_argument("--test-batch-size", default=1000, type=int,
                    help="test batch size.")


def load_network(args, seed=0, validation=False):
    if args.network == 'simple':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = SimpleCNN
    elif args.network == 'resnet':
        dataset = mpi_dataset.download_cifar10_retry(seed)
        network = ResNet
    elif args.network == 'lr':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = LinearRegression
    else:
        assert False
    return network(dataset=dataset,
                   batch_size=args.test_batch_size if validation else args.batch_size,)


@ray.remote
class ParameterServer(object):
    def __init__(self, args, two_phases=False):
        self.two_phases = two_phases
        self.net = load_network(args)
        self.quantizer = Quantizer(self.net.variables.placeholders)

    def apply_gradients(self, *gradients):
        decompressed = self.quantizer.decode(gradients)
        if self.two_phases:
            compressed = self.quantizer.encode(decompressed)
            decompressed = self.quantizer.decode([compressed])
        self.net.apply_gradients(decompressed)
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, args, worker_index):
        self.args = args
        self.worker_index = worker_index
        self.net = load_network(args, seed=worker_index)
        self.batch_size = args.batch_size
        self.dataset = self.net.dataset
        self.quantizer = Quantizer(self.net.variables.placeholders)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.dataset.train.next_batch(self.batch_size)
        return self.quantizer.encode(self.net.compute_gradients(xs, ys))

    def loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.test.next_batch(self.args.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_


if __name__ == "__main__":
    args = parser.parse_args()

    if args.quantizer.lower() == 'identical':
        Quantizer = IdenticalQuantizer
    elif args.quantizer.lower() == 'scalar':
        Quantizer = ScalarQuantizer
    elif args.quantizer.lower() == 'codebook':
        Quantizer = CodebookQuantizer
    else:
        assert False

    ray.init(redis_address=args.redis_address)

    ps = ParameterServer.remote(args, two_phases=args.two_phases)
    workers = [Worker.remote(args, worker_index)
               for worker_index in range(args.num_workers)]

    print("Iteration, time, loss, accuracy")
    i = 0
    timer = Timer()
    current_weights = ps.get_weights.remote()
    while True:
        for _ in range(10):
            gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
            current_weights = ps.apply_gradients.remote(*gradients)

            if i % 10 == 0:
                loss_accuracy = [
                    ray.get([worker.loss_accuracy.remote() for worker in workers])
                    for _ in range(args.test_batch_size//args.batch_size + 1)
                ]
                loss, accuracy = np.mean(np.array(loss_accuracy).reshape((-1, 2)), axis=0)
                print("%d, %.3f, %.3f, %.3f" % (i, timer.toc(), loss, accuracy))
            i += 1
