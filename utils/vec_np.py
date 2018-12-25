import numpy as np


def normalize(vecs, order=None):
    norms = np.linalg.norm(vecs, axis=1, ord=order)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(
        vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)
    # divide by zero problem
    return norms, normalized_vecs
