// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2017 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef KERNEL_GPU_H_
#define KERNEL_GPU_H_

#define EIGEN_USE_GPU

#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>



void GPUScaleAndAdd(int n, float scale1, float *x, float scale2, float *y, cudaStream_t stream);
void GPUScale(int n, float scaler, float *x, cudaStream_t stream);
void GPUAdd(int n, float *x, float *y, cudaStream_t stream);
void GPUFindMaxAndMin(float *array, float *maxandmin, int n, cudaStream_t stream);
curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream);
void GPUQuantizeValue(unsigned char *x, float *y, float *maxandmin, int n, curandState* states, cudaStream_t stream);
void GPUDequantizeValue(unsigned char *recv, float *maxandmin, float *x, int n, cudaStream_t stream);
void GPUCopyValue(float* x, float* y, int n, cudaStream_t stream);


void GPUFindMaxAndMin2(float *array, float *max, float *min, int *mutex, int n, cudaStream_t stream);

#endif
