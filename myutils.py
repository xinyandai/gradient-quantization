import time

import numpy as np


def normalize(vecs, order=None):
    norms = np.linalg.norm(vecs, axis=1, ord=order)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)
    # divide by zero problem
    return norms, normalized_vecs


class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.t_start = time.time()

    def __enter__(self):
        self.tic()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.t_start))

    def tic(self):
        self.t_start = time.time()

    def toc(self):
        return time.time() - self.t_start
