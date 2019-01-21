from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import mpi_dataset

import tensorflow as tf
import numpy as np

from scipy import stats
from vecs_io import fvecs_read
from myutils import normalize
from myutils import Timer

from model_resnet import ResNet
from model_modelc import ModelC
from model_simple import SimpleCNN
from model_lr import LinearRegression
from model_two_layer import TwoLayerNetwork


parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", required=True, type=int,
                    help="The number of workers to use.")
parser.add_argument("--two-phases", required=True, type=bool,
                    help="Using 2-phases quantization.")
parser.add_argument("--network", required=True, type=str,
                    help="Network architectures")
parser.add_argument("--batch-size", required=True, type=int,
                    help="batch size.")
parser.add_argument("--norm-level", required=True, type=int,
                    help="norm quantization level.")
parser.add_argument("--sub-dimension", required=True, type=int,
                    help="sub dimension.")
parser.add_argument("--num-codebook", required=True, type=int,
                    help="sub dimension.")
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
    elif args.network == 'resnet':
        dataset = mpi_dataset.download_cifar10_retry(seed, True)
        network = ResNet
    elif args.network == 'modelC':
        dataset = mpi_dataset.download_cifar10_retry(seed, True)
        network = ModelC
    else:
        assert False
    return network(dataset=dataset,
                   batch_size=args.test_batch_size if validation else args.batch_size,)


class IdenticalCompressor(object):
    def compress(self, gradient):
        return gradient

    def decompress(self, gradient):
        return gradient


class ScalarCompressor(object):
    def __init__(self, s, random):
        self.s = s
        self.random = random

    def compress(self, vec):
        lower_bound = np.min(vec)
        upper_bound = np.max(vec)
        scaled_vec = np.abs((vec - lower_bound) / (upper_bound - lower_bound)) * self.s
        l = np.array(scaled_vec).clip(0, self.s-1).astype(dtype=np.int32)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l
            l[:] += probabilities > np.random.uniform(0, 1, l.shape)

        return lower_bound, upper_bound, l

    def decompress(self, signature):
        [lower_bound, upper_bound, l] = signature
        scaled_vec = l.astype(dtype=np.float32)
        compressed = scaled_vec / self.s * (upper_bound - lower_bound) + lower_bound
        return compressed


class RandomCodebookCompressor(object):
    def __init__(self, size, shape, c_dim, k, s):
        self.Ks = k
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= c_dim else self.size
        self.code_dtype = np.uint32

        self.M = size // self.dim
        self.norm_quantizer = ScalarCompressor(s, True)
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)
        if self.dim == self.Ks:
            self.codewords = stats.ortho_group.rvs(self.dim).astype(np.float32)
        else:
            _, self.codewords = normalize(fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks)))
        self.c = self.codewords.T
        self.c_dagger = np.linalg.pinv(self.c)
        self.codewords = np.concatenate((self.codewords, -self.codewords))

    def compress(self, vec):
        vec = tf.reshape(vec, (-1, self.dim))
        bar_p = tf.transpose(tf.matmul(self.c_dagger, tf.transpose(vec)))
        l1_norms = tf.linalg.norm(bar_p, axis=1, ord=1)
        normalized_vecs = tf.div_no_nan(bar_p, l1_norms[:, tf.newaxis])

        tild_p = tf.clip_by_value(
            tf.concat((normalized_vecs, -normalized_vecs), axis=1), 0, 1)

        r = tf.random.uniform([tild_p.shape[0].value, 1], 0, 1)
        rs = tf.tile(r, (1, tild_p.shape[1].value))
        comp = tf.cumsum(tild_p, axis=1) > rs
        codes = tf.argmax(tf.cast(comp, tf.int32), axis=1)

        return [l1_norms, codes]  # TODO as type here

    def decompress(self, signature):
        [norms, codes] = signature

        norms = self.norm_quantizer.decompress(self.norm_quantizer.compress(norms))

        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)


class RandomCodebookQuantizer(object):
    def __init__(self, placeholders, c_dim, k, s):
        self.placeholders = placeholders
        self.layers = len(placeholders)
        self.pqs = [
            RandomCodebookCompressor(
                tf.reshape(v, [-1]).shape.as_list()[0],
                v.shape.as_list(), c_dim, k, s
            )
            if tf.reshape(v, [-1]).shape.as_list()[0] > 1000 else IdenticalCompressor()
            for _, v in placeholders.items()
        ]

        self.gradient_holders = [tf.placeholder(tf.float32, shape=v.shape) for _, v in placeholders.items()]
        self.codes = [pq.compress(g) for pq, g in zip(self.pqs, self.gradient_holders)]

    def encode(self, gradient, sess):
        feed_dict = {}
        for i in range(len(self.gradient_holders)):
            feed_dict[self.gradient_holders[i]] = gradient[i]
        return sess.run(self.codes, feed_dict=feed_dict)

    def decode(self, gradients):
        """
        :param gradients:
        :return:
        """
        for gradient in gradients:
            assert self.layers == len(gradient)

        aggregator = [None for _ in range(self.layers)]

        for gradient in gradients:
            for i, (pq, code) in enumerate(zip(self.pqs, gradient)):
                if aggregator[i] is None:
                    aggregator[i] = pq.decompress(code)
                else:
                    aggregator[i][:] += pq.decompress(code)
        for agg in aggregator:
            agg[:] = agg[:] / len(gradients)
        return aggregator


class Worker(object):
    def __init__(self, args, worker_index, two_phases=False):
        self.two_phases = two_phases
        self.args = args
        self.worker_index = worker_index
        with tf.Graph().as_default():
            self.net = load_network(args, seed=worker_index)
            self.batch_size = args.batch_size
            self.dataset = self.net.dataset
            self.quantizer = Quantizer(self.net.variables.placeholders, args.sub_dimension, args.num_codebook, args.norm_level)

    def compute_gradients(self):
        xs, ys = self.dataset.train.next_batch(self.batch_size)
        g = self.net.compute_gradients(xs, ys)
        return self.quantizer.encode(g, self.net.sess)

    def apply_gradients(self, gradients_):
        decompressed = self.quantizer.decode(gradients_)
        if self.two_phases:
            compressed = self.quantizer.encode(decompressed, self.net.sess)
            decompressed = self.quantizer.decode([compressed])
        self.net.apply_gradients(decompressed)

    def test_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.test.next_batch(self.args.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_

    def train_loss_accuracy(self):
        test_xs_, test_ys_ = self.dataset.valid.next_batch(self.args.batch_size)
        loss_, accuracy_ = self.net.compute_loss_accuracy(test_xs_, test_ys_)
        return loss_, accuracy_


if __name__ == "__main__":
    args = parser.parse_args()
    Quantizer = RandomCodebookQuantizer

    worker = Worker(args, 0, two_phases=args.two_phases)

    print("Iteration, time, loss, accuracy, train_loss, train_accuracy")
    i = 0
    timer = Timer()
    while True:
        for _ in range(100):

            gradients = [worker.compute_gradients() for _ in range(args.num_workers)]
            worker.apply_gradients(gradients)

            if i % 100 == 0:
                test_accuracy = [
                    worker.test_loss_accuracy() for _ in range(args.test_batch_size // args.batch_size + 1)
                ]
                train_accuracy = [
                    worker.train_loss_accuracy() for _ in range(args.test_batch_size // args.batch_size + 1)
                ]
                ts_loss, ts_accuracy = np.mean(np.array(test_accuracy).reshape((-1, 2)), axis=0)
                tr_loss, tr_accuracy = np.mean(np.array(train_accuracy).reshape((-1, 2)), axis=0)
                print("%d, %.3f, %.3f, %.3f, %.3f, %.3f" % (i, timer.toc(), ts_loss, ts_accuracy, tr_loss, tr_accuracy))
            i += 1
