#include <iostream>
#include "timer.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

__global__ void vectorAdd(const float *A, const float *B,float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
int main(int argc, char* argv[]) {

    int K = atoi(argv[1]);
    int numElements = K * 1000000; // Convert millions to actual size
    size_t size = numElements* sizeof(float);
    // printf("Vector addtion of %d element.\n",numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize arrays
    for (int i = 0; i<numElements; i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&h_A, size * sizeof(int));
    cudaMallocManaged(&h_B, size * sizeof(int));
    cudaMallocManaged(&h_C, size * sizeof(int));


    // Launch kernel on 1M elements on the GPU
    int threadsPerBlock = 256;
    int blocksPerGrid =  (numElements + threadsPerBlock-1)/threadsPerBlock;
    initialize_timer();
    start_timer();
    // Measure time taken to add arrays on GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, h_C, size);
    cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
    stop_timer();
    double time = elapsed_time();
    printf( "Time for %d million elements : %lf (sec)\n", K, time);

    // Free memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return 0;
}

// Time for 1 million elements : 0.042384 (sec)
// Time for 5 million elements : 0.013093 (sec)
// Time for 10 million elements : 0.020427 (sec)
// Time for 50 million elements : 0.073276 (sec)
// Time for 100 million elements : 0.140707 (sec)
