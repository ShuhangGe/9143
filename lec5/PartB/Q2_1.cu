#include<stdio.h>
#include<cuda_runtime.h>
#include "timer.h"

//A+b=C
__global__ void vectorAdd(const float *A, const float *B,float *C, int numElements)
{
    for (int i=0 ; i< numElements;i++)
    {
        C[i] = A[i] + B[i];
    }
}
int main(int argc, char* argv[])
{


    int K = atoi(argv[1]);
    int numElements = K * 1000000; // Convert millions to actual size
    size_t size = numElements* sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i<numElements; i++)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    //operate GPU kernel function
    int threadsPerBlock  = 1;
    int blockperGrid = 1;
    initialize_timer();
    start_timer();
    vectorAdd<<<blockperGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    stop_timer();
    double time = elapsed_time();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for(int i =0; i<numElements;i++)
    {
        if(fabs(h_A[i] + h_B[i] - h_C[i]>1e-5))
        {
            fprintf(stderr,"Result failed");
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    printf( "Time for %d million elements : %lf (sec)\n", K, time);

    return 0;

}


// Time for 1 million elements : 0.000359 (sec)
// Time for 5 million elements : 0.000253 (sec)
// Time for 10 million elements : 0.000251 (sec)
// Time for 50 million elements : 0.000292 (sec)
// Time for 100 million elements : 0.000291 (sec)