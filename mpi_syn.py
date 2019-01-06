from mpi_worker_vqsgd import CodebookQuantizerWorker
from mpi_worker_qsgd import ScalarQuantizerWorker
from mpi_worker import Worker
from myutils import Timer
from mpi4py import MPI
from model_simplenn import deepnn
from model_vgg import vgg_16
from model_alexnet import alexnet_v2

import mpi_model as model

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
    else:
        assert False

    if worker_index == 0:
        logging.info('initialize quantization worker as {}'.format(QuantizerWorker))

    if args.dataset == "mnist":
        dataset = model.download_mnist_retry(seed=worker_index)
    elif args.dataset == "cifar":
        dataset = model.download_cifar10_retry(seed=worker_index)
    else:
        assert False

    if args.network == "simple":
        net = deepnn
    elif args.network == "alexnet":
        net = alexnet_v2
    elif args.network == "vgg":
        net = vgg_16
    elif args.network == "resnet":
        assert False, "Not implemented yet"
    else:
        assert False

    nn = model.ModelCNN(dataset, net=net, learning_rate=1e-4,)
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
                logging.info("{}, {}, {}, {}".format(i, timer.toc(), loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
