import numpy as np
from scipy.cluster.vq import kmeans2
from utils.vecs_io import fvecs_writer


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    # divide by zero problem
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)
    return norms, normalized_vecs


def train_codebook(dimension, Ks, train_size=1000000, iter=20):
    X = np.random.normal(0, 1, size=(train_size, dimension))
    X = np.array(X, dtype=np.float32)
    _, X = normalize(X)

    codewords, _ = kmeans2(X, Ks, iter=iter, minit='points')
    return codewords


def generate():
    np.random.seed(808)
    for dim in range(1, 66):
        for Ks in [32, 64, 256, 512, 1024]:
            codebook = train_codebook(dim, Ks)
            assert codebook.shape == (Ks, dim)
            filename = 'codebook/angular_dim_{}_Ks_{}'.format(dim, Ks)
            fvecs_writer(filename, codebook)
            print('writing codebook into file {}'.format(filename))


if __name__ == '__main__':
    generate()
