import tensorflow as tf
from pymylibs import mylibs

mpi_ops = tf.load_op_library('ops/mpi_ops.so')


def mpi_initialize():
    return mylibs.mpi_initialize()


def mpi_finalize():
    return mylibs.mpi_finalize()


def mpi_rank():
    return mylibs.mpi_rank()


def mpi_size():
    return mylibs.mpi_size()


worker_size = mpi_size()
worker_rank = mpi_rank()


def mpi_ops_tf_gather(sendbuf, pre_node, root):
    pre_node = sendbuf if pre_node is None else pre_node
    return mpi_ops.tf_gather(sendbuf, pre_node, root=root, size=worker_size)


def mpi_ops_tf_broadcast(buff, pre_node, root):
    pre_node = buff if pre_node is None else pre_node
    return mpi_ops.tf_broadcast(buff, pre_node, root=root, size=worker_size)
