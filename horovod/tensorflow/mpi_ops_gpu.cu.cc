

#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "mpi_cuda.h"


#define maxThreadsPerBlock 1024


__global__ void _scaleAndAdd(int n, float scale1, float *x, float scale2, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = scale1 * x[i] + scale2 * y[i];              

}

__global__ void _scale(int n, float scaler, float *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = scaler * x[i];              

}

__global__ void _add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];              

}

__global__ void _findMaxAndMin2(float *array, float *max, float *min, int *mutex, int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache1[maxThreadsPerBlock];
    __shared__ float cache2[maxThreadsPerBlock];


    float temp1 = 0;
    float temp2 = 0;
    while(index + offset < n){

        temp1 = fmaxf(temp1, array[index + offset]);
        temp2 = fminf(temp2, array[index + offset]);

        offset += stride;
    }

    cache1[threadIdx.x] = temp1;
    cache2[threadIdx.x] = temp2;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){

            cache1[threadIdx.x] = fmaxf(cache1[threadIdx.x], cache1[threadIdx.x + i]);
            cache2[threadIdx.x] = fminf(cache2[threadIdx.x], cache2[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0){
        while(atomicCAS(mutex,0,1) != 0);  //lock
        *max = fmaxf(*max, cache1[0]);
        *min = fminf(*min, cache2[0]);
        atomicExch(mutex, 0);  //unlock
    }
}

__global__ void _findMaxAndMin(float *array, float *maxandmin, int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    __shared__ float cache1[maxThreadsPerBlock];
    __shared__ float cache2[maxThreadsPerBlock];

    for(int j = index; j < n; j += stride)
    {

        int my_bucket = j / 512;
        int index_in_bucket = j % 512;
        int offset = (my_bucket&1) ? 512 : 0;

        // reduction
        unsigned int i = 512 / 2;
        while(i != 0)
        {
            if(index_in_bucket < i)
            {

                if(i == 512 / 2) //get data in cache in first loop
                {
                    cache1[index_in_bucket + offset] = fmaxf(array[j], array[j + i]);
                    cache2[index_in_bucket + offset] = fminf(array[j], array[j + i]);                 
                }
                else
                {
                    cache1[index_in_bucket + offset] = fmaxf(cache1[index_in_bucket + offset], cache1[index_in_bucket + offset + i]);
                    cache2[index_in_bucket + offset] = fminf(cache2[index_in_bucket + offset], cache2[index_in_bucket + offset + i]);  
                }

            }
            __syncthreads();
            i /= 2;
        }

        

        if(threadIdx.x == 0)
        {
            maxandmin[2 * my_bucket] = cache1[0];
            maxandmin[2 * my_bucket + 1] = cache2[0];
        }
        else if(threadIdx.x == 512)
        {
            maxandmin[2 * my_bucket] = cache1[512];
            maxandmin[2 * my_bucket + 1] = cache2[512];
        }
    }

}

__global__ void _initCURand(unsigned int seed, curandState* states)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              index, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
}


__global__ void _quantizeValue(unsigned char *x, const float *y, const float *maxandmin, const int n, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    curandState local_state;
    local_state = states[index];


    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / 512;
        float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / 255.0;
        float d = (y[i] - maxandmin[my_bucket * 2 + 1]) / unit + (curand(&local_state)%1000001 / 1000000.0); 
        x[i] = (unsigned char) floor(d);
    }
    states[index] = local_state;       
}




__global__ void _dequantizeValue(unsigned char *recv, float *maxandmin, float *x, const int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / 512;
        float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / 255.0;
        x[i] = maxandmin[my_bucket * 2 + 1] + recv[i] * unit;  
    }          
}

__global__ void _copyValue(float* x, const float* y, const int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = y[i];
}




void GPUScaleAndAdd(int n, float scale1, float *x, float scale2, float *y, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _scaleAndAdd<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, scale1, x, scale2, y);
    cudaStreamSynchronize(stream);	    
}

void GPUScale(int n, float scaler, float *x, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _scale<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, scaler, x);
    cudaStreamSynchronize(stream);	    
}

void GPUAdd(int n, float *x, float *y, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _add<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, x, y);
    cudaStreamSynchronize(stream);	    
}


void GPUFindMaxAndMin(float *array, float *maxandmin, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _findMaxAndMin<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(array, maxandmin, n);
    cudaStreamSynchronize(stream); 
}

void GPUFindMaxAndMin2(float *array, float *max, float *min, int *mutex, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _findMaxAndMin2<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(array, max, min, mutex, n);
    cudaStreamSynchronize(stream);
    
}

curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream)
{
    curandState* states;

    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    cudaMalloc(&states, blocksPerGrid * maxThreadsPerBlock * sizeof(curandState));

    _initCURand<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(seed, states);

    return states;    
}

void GPUQuantizeValue(unsigned char *x, float *y, float *maxandmin, int n, curandState* states, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _quantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, maxandmin, n, states);
    cudaStreamSynchronize(stream);
    
}

void GPUDequantizeValue(unsigned char *recv, float *maxandmin, float *x, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _dequantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(recv, maxandmin, x, n);
    cudaStreamSynchronize(stream);
    
}


void GPUCopyValue(float* x, float* y, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _copyValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, n);
    cudaStreamSynchronize(stream);
    
} 
