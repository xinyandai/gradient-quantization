import tensorflow as tf

import numpy as np
import mpi_wrappers

from tf_variables import unflatten
from distributed_optimizer import DistributedOptimizer

mpi_ops = tf.load_op_library('ops/mpi_ops.so')

worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class DistributedCompressorOptimizer(DistributedOptimizer):
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        super(DistributedCompressorOptimizer, self).__init__(
            optimizer, name=name, use_locking=use_locking,
            device_dense=device_dense, device_sparse=device_sparse)

    def synchronize_grads(self, gradients):
        averaged_gradients = []
        with tf.name_scope(self._name + "_Allreduce"):
            pre_operation = None
            gathered_tensors = []
            with tf.device(self._device_dense):
                for layer, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        root_rank = layer % worker_size
                        pre_operation = mpi_wrappers.mpi_ops_tf_gather(grad, pre_operation, root=root_rank)
                        gathered_tensors.append(pre_operation)

                averaged = [tf.reduce_sum(t, 0) / worker_size for t in gathered_tensors]

                for layer, (gradient, var) in enumerate(gradients):
                    if gradient is not None:
                        root_rank = layer % worker_size
                        pre_operation = mpi_wrappers.mpi_ops_tf_broadcast(averaged[layer], pre_operation, root=root_rank)
                        averaged_gradients.append((pre_operation, var))
                    else:
                        averaged_gradients.append((None, var))
        return averaged_gradients
