from mpi_worker_vqsgd import CodebookQuantizerWorker
from mpi_worker_qsgd import ScalarQuantizerWorker
from mpi_worker import Worker
from myutils import Timer
import logging
import argparse

parser = argparse.ArgumentParser(description="Run the synchronous All-Reduce example.")
parser.add_argument("--quantizer", default='codebook', type=str,
                    help="Compressor for gradient.")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if args.quantizer.lower() == 'identical':
        QuantizerWorker = Worker
    elif args.quantizer.lower() == 'scalar':
        QuantizerWorker = ScalarQuantizerWorker
    elif args.quantizer.lower() == 'codebook':
        QuantizerWorker = CodebookQuantizerWorker
    else:
        assert False
    logging.info('initialize quantization worker as {}'.format(QuantizerWorker))
    worker = QuantizerWorker()
    worker.syn_weights(worker.net.variables.get_flat())

    logging.info("Iteration, time, loss, accuracy")
    timer = Timer()
    i = 0
    while i <= 200:
        if i % 10 == 0:
            # Evaluate the current model.
            loss, accuracy = worker.compute_loss_accuracy()
            logging.info("{}, {}, {}, {}".format(i, timer.toc(), loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
