from mpi_worker_vqsgd import CodebookQuantizerWorker
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('initialize worker')
    worker = CodebookQuantizerWorker()
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
