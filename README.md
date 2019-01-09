#  quantization for stochastic gradient
* quantization methods
    - identical quantization
    - scalar quantization
    - codebook quantization
* implementation 
    - Ray 
    
    python ray_syn.py --quantizer codebook --num-workers 4
    - MPi 
    
    mpirun -np 4 python mpi_syn.py --quantizer codebook --dataset mnist --network simple
    
    -cuda-aware MPI(developing)
    
    