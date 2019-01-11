from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
import ray_model as model
from quantizer_identical import IdenticalQuantizer
from quantizer_scalar import ScalarQuantizer
from quantizer_codebook import CodebookQuantizer


Quantizer = None
parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--quantizer", default='codebook', type=str,
                    help="Compressor for gradient.")
parser.add_argument("--two-phases", default=False, type=bool,
                    help="Using 2-phases quantization.")


@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate, two_phases=False):
        self.two_phases = two_phases
        self.net = model.SimpleCNN(learning_rate=learning_rate)
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
    def __init__(self, worker_index, batch_size=50):
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.mnist = model.download_mnist_retry(seed=worker_index)
        self.net = model.SimpleCNN()
        self.quantizer = Quantizer(self.net.variables.placeholders)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        return self.quantizer.encode(self.net.compute_gradients(xs, ys))


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

    # Create a parameter server.
    net = model.SimpleCNN()

    ps = ParameterServer.remote(1e-2 * args.num_workers, two_phases=args.two_phases)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()

    i = 0
    current_weights = ps.get_weights.remote()

    from myutils import Timer
    timer = Timer()
    print("Iteration, time, loss, accuracy")
    while True:
        # Compute and apply gradients.
        for _ in range(10):
            gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
            current_weights = ps.apply_gradients.remote(*gradients)

            if i % 10 == 0:
                # Evaluate the current model.
                net.variables.set_flat(ray.get(current_weights))
                test_xs, test_ys = mnist.test.next_batch(1000)
                loss, accuracy = net.compute_loss_accuracy(test_xs, test_ys)
                print("%d, %.3f, %.3f, %.3f" % (i, timer.toc(), loss, accuracy))
            i += 1
