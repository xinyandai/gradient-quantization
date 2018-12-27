from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
import ray_model as model
from quantizer_identical import IdenticalQuantizer
from quantizer_scalar import ScalarQuantizer


Quantizer = None
parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")
parser.add_argument("--quantizer", default='identical', type=str,
                    help="Compressor for gradient.")



@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = model.SimpleCNN(learning_rate=learning_rate)
        self.quantizer = Quantizer(self.net.variables.placeholders)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(self.quantizer.decode(gradients))
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
    else:
        assert False

    ray.init(redis_address=args.redis_address)

    # Create a parameter server.
    net = model.SimpleCNN()

    ps = ParameterServer.remote(1e-4 * args.num_workers)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()

    i = 0
    current_weights = ps.get_weights.remote()

    while True:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % 10 == 0:
            # Evaluate the current model.
            net.variables.set_flat(ray.get(current_weights))
            test_xs, test_ys = mnist.test.next_batch(1000)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print("Iteration {}: accuracy is {}".format(i, accuracy))
        i += 1
