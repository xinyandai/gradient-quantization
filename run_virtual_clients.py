from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import mpi_dataset

import numpy as np

from myutils import Timer

from model_resnet import ResNet
from model_modelc import ModelC
from model_simple import SimpleCNN
from model_lr import LinearRegression
from model_two_layer import TwoLayerNetwork

from quantizer_identical import IdenticalQuantizer
from quantizer_scalar import ScalarQuantizer
from quantizer_codebook import CodebookQuantizer
from quantizer_random_codebook import RandomCodebookQuantizer

from models import FCN, CNN


Quantizer = None
parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", required=True, type=int,
                    help="The number of workers to use.")
parser.add_argument("--quantizer", required=True, type=str,
                    help="Compressor for gradient.")
parser.add_argument("--two-phases", required=True, type=bool,
                    help="Using 2-phases quantization.")
parser.add_argument("--network", required=True, type=str,
                    help="Network architectures")
parser.add_argument("--batch-size", required=True, type=int,
                    help="batch size.")
parser.add_argument("--test-batch-size", default=1024, type=int,
                    help="test batch size.")


def load_network(args, seed=0, validation=False):
    if args.network == 'simple':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = SimpleCNN
    elif args.network == 'lr':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = LinearRegression
    elif args.network == 'two_layer':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = TwoLayerNetwork
    elif args.network == 'fcn':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = FCN
    elif args.network == 'cnn':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = CNN
    elif args.network == 'resnet':
        dataset = mpi_dataset.download_cifar10_retry(seed)
        network = ResNet
    elif args.network == 'modelC':
        dataset = mpi_dataset.download_cifar10_retry(seed)
        network = ModelC
    else:
        assert False
    return network(dataset=dataset,
                   batch_size=args.test_batch_size if validation else args.batch_size,)


class Worker(object):
    def __init__(self, args, worker_index, two_phases=False):
        self.two_phases = two_phases
        self.args = args
        self.worker_index = worker_index
        self.net = load_network(args, seed=worker_index)
        self.batch_size = args.batch_size
        self.dataset = self.net.dataset
        self.quantizer = Quantizer(self.net.variables.placeholders)

    def compute_gradients(self):
        xs, ys = self.dataset.train.next_batch(self.batch_size)
        g = self.net.compute_gradients(xs, ys)
        return self.quantizer.encode(g)

    def apply_gradients(self, gradients_):
        decompressed = self.quantizer.decode(gradients_)
        if self.two_phases:
            compressed = self.quantizer.encode(decompressed)
            decompressed = self.quantizer.decode([compressed])
        self.net.apply_gradients(decompressed)

    def test_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.test.next_batch(self.args.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_

    def train_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.valid.next_batch(
            self.args.batch_size)
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
    elif args.quantizer.lower() == 'random_codebook':
        Quantizer = RandomCodebookQuantizer
    else:
        assert False

    worker = Worker(args, 0, two_phases=args.two_phases)

    print("Iteration, time, loss, accuracy, train_loss, train_accuracy")
    i = 0
    timer = Timer()
    while True:
        gradients = [worker.compute_gradients()
                     for _ in range(args.num_workers)]
        worker.apply_gradients(gradients)

        if i % 10 == 0:
            test_accuracy = [
                worker.test_loss_accuracy() for _ in range(args.test_batch_size // args.batch_size + 1)
            ]
            train_accuracy = [
                worker.train_loss_accuracy() for _ in range(args.test_batch_size // args.batch_size + 1)
            ]
            ts_loss, ts_accuracy = np.mean(
                np.array(test_accuracy).reshape((-1, 2)), axis=0)
            tr_loss, tr_accuracy = np.mean(
                np.array(train_accuracy).reshape((-1, 2)), axis=0)
            print("%d, %.3f, %.3f, %.3f, %.3f, %.3f" %
                  (i, timer.toc(), ts_loss, ts_accuracy, tr_loss, tr_accuracy))
        i += 1
