from mpi_worker_vqsgd import CodebookQuantizerWorker
from mpi_worker_qsgd import ScalarQuantizerWorker
from mpi_worker import Worker
import logging
import argparse

parser = argparse.ArgumentParser(description="Run the synchronous All-Reduce example.")
parser.add_argument("--quantizer", default='codebook', type=str,
                    help="Compressor for gradient.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('initialize worker')
    args = parser.parse_args()

    if args.quantizer.lower() == 'identical':
        QuantizerWorker = Worker
    elif args.quantizer.lower() == 'scalar':
        QuantizerWorker = ScalarQuantizerWorker
    elif args.quantizer.lower() == 'codebook':
        QuantizerWorker = CodebookQuantizerWorker
    else:
        assert False
    worker = QuantizerWorker()
    logging.debug('synchronize weights before running.')
    worker.syn_weights(worker.net.variables.get_flat())
    i = 0
    while i <= 200:
        if i % 10 == 0:
            # Evaluate the current model.
            loss, accuracy = worker.compute_loss_accuracy()
            logging.warning("Iteration {}: loss is {}, accuracy is {}".format(i, loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
