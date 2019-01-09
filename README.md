#  quantization for stochastic gradient
## quantization methods
* identical quantization
* scalar quantization
* codebook quantization
## implementation 
Before Run

    pip install mpi4py
    pip install ray
    pip install tensorflow
Run Ray

    python ray_syn.py --quantizer codebook --num-workers 4
Run MPI

    mpirun -np 4 python mpi_syn.py --quantizer codebook --dataset mnist --network simple
Run cuda-aware MPI(developing)

    git clone https://github.com/xinyandai/similarity-search.git
    cd similarity-search
    sh srp.sh