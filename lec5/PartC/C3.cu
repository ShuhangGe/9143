#include <iostream>
#include <vector>
#include <cudnn.h>
#include <cuda_runtime.h>

float CheckSum(double* M) {
  float result = 0;
  for(int k = 0; k < K; k++) {
    for (int i = 0; i < W; ++i) {
      for (int j = 0; j < H; ++j) {
        result += M[k * W * H + i * H + j];
      }
    }
  }
  return result;
}

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define tensor sizes and filter dimensions
    // Example: input = 1x3x128x128, filter = 10x3x3x3
    int batch_size = 1, channels = 3, height = 1024, width = 1024;
    int filter_height = 3, filter_width = 3, output_channels = 64;

    // Create and set descriptors
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_channels, channels, filter_height, filter_width);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Find the dimensions of the output tensor
    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &n, &c, &h, &w);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    // Allocate memory for input, output, and filter
    float *input, *output, *filter;
    cudaMalloc(&input, batch_size * channels * height * width * sizeof(float));
    cudaMalloc(&output, n * c * h * w * sizeof(float));
    cudaMalloc(&filter, output_channels * channels * filter_height * filter_width * sizeof(float));

    // Initialize memory for input and filter (omitted for brevity)

    // Selecting the convolution algorithm (CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
    cudnnConvolutionFwdAlgo_t algo = convolution_algorithm.algo;

    int returnedAlgoCount;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, 1, &returnedAlgoCount, &convolution_algorithm);

    // Allocate workspace for the selected algorithm
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, algo, &workspace_bytes);
    void *workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&workspace, workspace_bytes);
    }

    // Perform the convolution
    float alpha = 1.0f, beta = 0.0f;

    initialize_timer();
    start_timer();
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input, filter_descriptor, filter, convolution_descriptor, algo, workspace, workspace_bytes, &beta, output_descriptor, output);
    stop_timer();

    double time = elapsed_time();
    float checkSum = CheckSum(output);
    // float checkSum = 0;
    printf( "%lf, %lf, %lf\n", checkSum, time*1000);
    
    cudaError_t cudaStatus = cudaGetLastError();
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error accordingly
    }
    // Print the kernel execution time
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Compute checksum (sum of elements in 'output')
    // ...

    // Cleanup
    if (workspace) cudaFree(workspace);
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
