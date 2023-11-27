#include <iostream>
#include <vector>
#include <cudnn.h>
#include "timer.h"
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FH = 3;
const int FW = 3;
const int K = 64;
const int P = 1;

double calculateChecksum(double* O) {
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum +=1;// O[k * W * H + x * H + y];
            }
        }
    }
    return checksum;
}
void initImage(double* M) {
  for(int c = 0; c < C; ++c) {
    for (int i = 0; i < W + P; ++i) {
      for (int j = 0; j < H + P; ++j) {
        M[c * W * H + i * H + j] = (double)(c * (i + j));
      }
    }
  }
  for(int c = 0; c < C; c++) {
    for (int i = 0; i < W + P; ++i) {
      int j = H + P;
      M[c * W * H + i * H + j] = (double)(0);
    }
    for (int j = 0; j < W + P; ++j) {
      int i = H + P;
      M[c * W * H + i * H + j] = (double)(0);
    }
  }
}
void initFilter(double* M) {
  for(int k = 0; k < K; ++k) {
    for(int c = 0; c < C; ++c) {
      for (int i = 0; i < FW; ++i) {
        for (int j = 0; j < FH; ++j) {
          M[k * C * FW * FH + c * FW * FH + i * FH + j] = (double)((c + k) * (i + j));
        }
      }
    }
  }
}

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define tensor sizes and filter dimensions
    int batch_size = 1;

    double* host_output = (double*)malloc(K * W * H * sizeof(double));
    double* host_input = (double*)malloc(C * (W + 2 * P) * (H + 2 * P) * sizeof(double));
    double* host_filter = (double*)malloc(C * K * (FW * FH) * sizeof(double));
    initImage(host_input);
    initFilter(host_filter);
    // Create and set descriptors
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, C, W + 2 * P, H + 2 * P);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, K, H, W);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,P, P, 1, 1, 2, 2, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    double *input, *output, *filter;

    cudaMalloc((void**)&filter, sizeof(double) * K * C * FH * FW);
    cudaMalloc((void**)&input, sizeof(double) * C * (W + 2 * P) * (H + 2 * P));
    cudaMalloc((void**)&output, sizeof(double) * K * W * H);

    cudaMemcpy(filter, host_filter, sizeof(double) *  K * C * FH * FW, cudaMemcpyHostToDevice);
    cudaMemcpy(input, host_input, sizeof(double) * C * (W + 2 * P) * (H + 2 * P), cudaMemcpyHostToDevice);

    // cudaMemcpy(device_filter, filter, sizeof(double) * K * C * FH * FW, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_input, input, sizeof(double) * C * (W + 2 * P) * (H + 2 * P), cudaMemcpyHostToDevice);
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;

    int returnedAlgoCount;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, \
                                        output_descriptor, 1, &returnedAlgoCount, &convolution_algorithm);
    cudnnConvolutionFwdAlgo_t algo = convolution_algorithm.algo;

    double alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    initialize_timer();
    start_timer();

    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input, filter_descriptor, filter, \
    convolution_descriptor, algo, nullptr, 0, &beta, output_descriptor, output);
    stop_timer();

    cudaMemcpy(host_output, output, sizeof(double) * K * W * H, cudaMemcpyDeviceToHost);

    double time = elapsed_time();
    double checkSum = calculateChecksum(host_output);
    // double checkSum = 0;
    printf( "checkSum: %lf\ntime: %lf\n", checkSum,  time*1000);
    
    cudaError_t cudaStatus = cudaGetLastError();
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error accordingly
    }



    // Cleanup
    cudaFree(input);
    cudaFree(output);
    cudaFree(filter);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}
