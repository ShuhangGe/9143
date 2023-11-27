// vecAddKernel00.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// This Kernel adds two Vectors A and B in C on GPU
// without using coalesced memory access.
#include <stdio.h>

__global__ void AddVectors01(const float* A, const float* B, float* C, int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
                i < N * gridDim.x * blockDim.x; i += gridDim.x * blockDim.x) {
            // printf("i = %d\n", i);
            C[i] = A[i] + B[i];
        }
    
}
