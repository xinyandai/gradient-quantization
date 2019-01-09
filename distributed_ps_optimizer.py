import tensorflow as tf
import numpy as np

import logging
import mpi_wrappers

from tf_variables import unflatten
from distributed_optimizer import DistributedOptimizer


worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


def synchronize_grad(tensor, root_rank, pre_node=None):
    """pre_node is only used to guarantee that start broadcast tensor after pre_node complete"""
    if pre_node is None:
        pre_node = tensor
    gathered_tensor = mpi_wrappers.mpi_ops_tf_gather(tensor, pre_node, root=root_rank)
    averaged_tensor = tf.reduce_sum(gathered_tensor, 0) / worker_size
    return mpi_wrappers.mpi_ops_tf_broadcast(averaged_tensor, pre_node, root=root_rank)


class DistributedPSOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        super(DistributedPSOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def synchronize_grads(self, gradients):
        averaged_gradients = []
        with tf.name_scope(self._name + "_Allreduce"):
            pre_operation = None
            for layer, (grad, var) in enumerate(gradients):
                if grad is not None:
                    with tf.device(self._device_dense):
                        updated_tensor = synchronize_grad(grad, layer % worker_size, pre_operation)
                        averaged_gradients.append((updated_tensor, var))
                        pre_operation = updated_tensor
                else:
                    averaged_gradients.append((None, var))
        return averaged_gradients
