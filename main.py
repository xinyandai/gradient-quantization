from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import numpy as np

from myutils import Timer
import mpi_dataset
from model_resnet import ResNet
from model_modelc import ModelC

from models import FCN, CNN, LinearRegression
from quantizers import IdenticalQuantizer, ScalarQuantizer, CodebookQuantizer, RandomCodebookQuantizer
from worker import Worker

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


def main():
    args = parser.parse_args()
    network = load_network(args)
    quantizer = load_quantizer(args, network)
    worker = Worker(network, quantizer, args.batch_size,
                    two_phases=args.two_phases)

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


def load_network(args, seed=0, validation=False):
    if args.network == 'lr':
        dataset = mpi_dataset.download_mnist_retry(seed)
        network = LinearRegression
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


def load_quantizer(args, network):
    Quantizer = None
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
    return Quantizer(network.variables.placeholders)


if __name__ == "__main__":
    main()
