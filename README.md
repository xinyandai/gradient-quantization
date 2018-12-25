# Setup QSGD-TF on Linux
This tutorial demonstrates how to install and use QSGD-TF in the distributed cluster. [Horovod](https://github.com/uber/horovod) is a distributed training framework for TensorFlow. It adopts data parallelism and MPI communication to scale Tensorflow in clusters  more efficiently than original distributed Tensorflow did. In this project, we have implemented Quantized-SGD based on Horovod in order to further reduce the distributed training time.

Below is a figure representing the benchmark that was done on 32 Amazon EC2 p2.xlarge instances with 1 NVIDIA K80 GPU each. We have trained four popular CNN models on Imagenet and compared them with original Horovod implementation. The figure shows epoch time on 8, 16, 32 GPUs, for full 32-bit precision of Tensorflow versus QSGD 8-bit. Epoch time is broken down into communication (bottom solid color) and computation (top transparent color):

![Results](/docs/Results.png)

### Precondition 
1. If you use GPU cluster, you should install [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) first.
2. Install [Tensorflow](https://www.tensorflow.org/install/).
3. Install [OpenMPI](https://www.open-mpi.org/) or another MPI implementation. Steps to install OpenMPI are listed [here](https://www.open-mpi.org/faq/?category=building#easy-build).
 **Attention :** For GPU cluster, you should make sure your MPI is compiled with **cuda-aware**, which can support the GPU communication with MPI. You can get more details of cuda-aware OpenMPI [here](https://www.open-mpi.org/faq/?category=buildcuda). You can use following command to see if your MPI was built with CUDA-aware support.
    ```sh
    $ ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    $ mca:mpi:base:param:mpi_built_with_cuda_support:value:true
    ```
### Compile QSGD-TF 
 1. Compile QSGD-TF with `pip`: (This `pip` should be in the same python environment as your Tensorflow.)
    ```sh
    $ HOROVOD_GPU_ALLREDUCE=MPI pip install /path/to/QSGD-TF/folder/
    ```
    Here we do not need NCCL flag as Horovod dose. Horovod uses NCCL to do collective communications like Allreduce or Broadcast, but we just use basic MPI operations. 
    
### Usage of QSGD-TF

To use Horovod, make the following additions to your program:

1. Run `hvd.init()`.

2. Pin a server GPU to be used by this process using `config.gpu_options.visible_device_list`. With the typical setup of one GPU per process, this can be set to local rank. In that case, the first process on the server will be allocated the first GPU, second process will be allocated the second GPU and so forth.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by the number of workers. An increase in learning rate compensates for the increased batch size.

4. Wrap optimizer in `hvd.DistributedOptimizer`. The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using allreduce or allgather, and then applies those averaged gradients.

5. Add `hvd.BroadcastGlobalVariablesHook(0)` to broadcast initial variable states from rank 0 to all other processes. This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint. Alternatively, if you're not using `MonitoredTrainingSession`, you can simply execute the `hvd.broadcast_global_variables` op after global variables have been initialized.

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them. This can be accomplished by passing `checkpoint_dir=None` to `tf.train.MonitoredTrainingSession` if `hvd.rank() != 0`.

You can find full training examples in the project: `/QSGD_TF/examples`.

### Running QSGD-TF
The examples below are for Open MPI. Check your MPI documentation for arguments to the `mpirun` command on your system.

Typically one GPU will be allocated per process, so if a server has 4 GPUs, you would run 4 processes. In Open MPI, the number of processes is specified with the `-np` flag.

Starting with the Open MPI 3, it's important to add the `-bind-to none` and `-map-by slot` arguments. `-bind-to none` specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance). `-map-by slot` allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.

`-mca pml ob1` and `-mca btl ^openib` flags force the use of TCP for MPI communication. This avoids many multiprocessing issues that Open MPI has with RDMA which typically result in segmentation faults.

With the `-x` option you can specify or copy (-x LD_LIBRARY_PATH) an environment variable to all the workers.

1. To run on a machine with 4 GPUs:
    ```sh
    $ mpirun -np 4 \
        -H localhost:4 \
        -bind-to none -map-by slot \
        -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python train.py
    ```
2. To run on 4 machines with 4 GPUs each:
    ```sh
    $ mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python train.py
    ```
The host where `mpirun` is executed must be able to SSH to all other hosts without any prompts.
If `mpirun` hangs without any output, verify that you can ssh to every other server without entering a password or answering questions.
### Troubleshooting
More FAQ can be found at `/QSGD_TF/docs/troubleshooting.md`.
Also you can find more useful information about [Horovod](https://github.com/uber/horovod) at Github.

