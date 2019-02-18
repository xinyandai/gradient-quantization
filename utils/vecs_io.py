import numpy as np
import struct


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


# we mem-map the biggest files to avoid having them in memory all at
# once
def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def bvecs_read(filename):
    return mmap_bvecs(fname=filename)


def fvecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def loader(data_set='audio', top_k=20, ground_metric='euclid', folder='../data/', verbose=True):
    """
    :param data_set: data set you wanna load , audio, sift1m, ..
    :param top_k: how many nearest neighbor in ground truth file
    :param ground_metric:
    :param folder:
    :return: X, Q, G
    """
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query.fvecs' % data_set
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, ground_metric)

    if verbose:
        print("# load the base data {}, \n".format(base_file),
              "# load the queries {}, \n",format(query_file),
              "# load the ground truth {}".format(ground_truth)
              )
    X = fvecs_read(base_file)
    Q = fvecs_read(query_file)
    G = ivecs_read(ground_truth)
    return X, Q, G


def bvecs_loader(data_set, top_k, ground_metric, folder='../data/'):
    """
    :param data_set: data set you wanna load , audio, sift1m, ..
    :param top_k: how many nearest neighbor in ground truth file
    :param ground_metric:
    :param folder:
    :return: X, Q, G
    """
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query.fvecs' % data_set
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, ground_metric)

    print("# load the base data {}, \n# load the queries {}, \n# load the ground truth {}".format(base_file, query_file,
                                                                                            ground_truth))
    X = bvecs_loader(base_file)
    Q = fvecs_read(query_file)
    G = ivecs_read(ground_truth)
    return X, Q, G

