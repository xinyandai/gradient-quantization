import tensorflow as tf
import mpi_wrappers
import logging

mpi_ops = tf.load_op_library('ops/mpi_ops.so')

worker_index = mpi_wrappers.mpi_rank()
worker_size = mpi_wrappers.mpi_size()


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(self, root_rank=0, device=''):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.b_cast_op = None
        self.device = device

    def begin(self):
        if self.b_cast_op is None:
            self.b_cast_op = [
                mpi_ops.tf_broadcast(v, v, root=0, size=worker_size)
                for v in tf.global_variables()
            ]

    def after_create_session(self, session, coord):
        self.synchronize_variables(session=session)

    def synchronize_variables(self, session):
        self.begin()
        for i, op in enumerate(self.b_cast_op):
            session.run(op)


class DistributedOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse

        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def synchronize_grads(self, gradients):
        raise NotImplementedError()

    def compute_gradients(self, *args, **kwargs):
        gradients = (super(DistributedOptimizer, self).compute_gradients(*args, **kwargs))
        logging.info("worker_size: {}".format(worker_size))
        if worker_size > 1:
            return self.synchronize_grads(gradients)
        else:
            return gradients

    def _apply_dense(self, *args, **kwargs):
        return self._optimizer._apply_dense(*args, **kwargs)

    def _resource_apply_dense(self, *args, **kwargs):
        return self._optimizer._resource_apply_dense(*args, **kwargs)

    def _resource_apply_sparse_duplicate_indices(self, *args, **kwargs):
        return self._optimizer._resource_apply_sparse_duplicate_indices(*args, **kwargs)

    def _resource_apply_sparse(self, *args, **kwargs):
        return self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _apply_sparse_duplicate_indices(self, *args, **kwargs):
        return self._optimizer._apply_sparse_duplicate_indices(*args, **kwargs)

    def _apply_sparse(self, *args, **kwargs):
        return self._optimizer._apply_sparse(*args, **kwargs)

    def _prepare(self, *args, **kwargs):
        return self._optimizer._prepare(*args, **kwargs)

    def _create_slots(self, *args, **kwargs):
        return self._optimizer._create_slots(*args, **kwargs)

    def _valid_dtypes(self, *args, **kwargs):
        return self._optimizer._valid_dtypes(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        return self._optimizer._finish(*args, **kwargs)