from mpi_worker_vqsgd import CodebookQuantizerWorker
from mpi_worker_polytope import PolytopeQuantizerWorker
from mpi_worker_qsgd import ScalarQuantizerWorker
from mpi_worker import Worker
from myutils import Timer
from mpi4py import MPI
from model_simple import SimpleCNN
from model_resnet import ResNet

import mpi_dataset

import logging
import argparse

parser = argparse.ArgumentParser(description="Run the synchronous All-Reduce example.")
parser.add_argument("--quantizer", type=str, required=True, help="Compressor for gradient, codebook/identical/scalar")
parser.add_argument("--dataset", type=str, required=True, help="Dataset for train, mnist/cifar10")
parser.add_argument("--network", type=str, required=True, help="Network for train, simple/alexnet/vgg")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    comm = MPI.COMM_WORLD
    worker_index = comm.Get_rank()
    worker_size = comm.Get_size()

    args = parser.parse_args()

    if args.quantizer.lower() == 'identical':
        QuantizerWorker = Worker
    elif args.quantizer.lower() == 'scalar':
        QuantizerWorker = ScalarQuantizerWorker
    elif args.quantizer.lower() == 'codebook':
        QuantizerWorker = CodebookQuantizerWorker
    elif args.quantizer.lower() == 'polytope':
        QuantizerWorker = PolytopeQuantizerWorker
    else:
        assert False

    if worker_index == 0:
        logging.info('initialize quantization worker as {}'.format(QuantizerWorker))

    if args.dataset == "mnist":
        dataset = mpi_dataset.download_mnist_retry(seed=worker_index)
    elif args.dataset == "cifar":
        dataset = mpi_dataset.download_cifar10_retry(seed=worker_index)
    else:
        assert False

    if args.network == "simple":
        nn = SimpleCNN(dataset)
    elif args.network == "resnet":
        nn = ResNet(dataset)
    else:
        assert False

    worker = QuantizerWorker(net=nn, dataset=dataset, lr=1e-4)

    worker.syn_weights(worker.net.variables.get_flat())

    if worker_index == 0:
        logging.info("Iteration, time, loss, accuracy")

    timer = Timer()
    i = 0
    while i <= 2000:
        if i % 10 == 0:
            # Evaluate the current model.
            loss, accuracy = worker.compute_loss_accuracy()
            if worker_index == 0:
                logging.info("%d, %.3f, %.3f, %.3f" % (i, timer.toc(), loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
