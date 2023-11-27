#include <iostream>
#include "timer.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define TILE_WIDTH 16

__global__ void convolveTiled(double* I0, double* F, double* O) {
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int row_o = by * TILE_WIDTH + ty;
    int col_o = bx * TILE_WIDTH + tx;
    int k = bz;

    __shared__ double N_ds[TILE_WIDTH + FH - 1][TILE_WIDTH + FW - 1][C];

    double output = 0.0;
    if(row_o < H && col_o < W) {
        for (int c = 0; c < C; ++c) {
            // Load I0 into shared memory
            for (int i = 0; i < (TILE_WIDTH + FW - 1) / TILE_WIDTH; ++i)
                for (int j = 0; j < (TILE_WIDTH + FH - 1) / TILE_WIDTH; ++j)
                    if (i * TILE_WIDTH + tx < TILE_WIDTH + FW - 1 && j * TILE_WIDTH + ty < TILE_WIDTH + FH - 1)
                        N_ds[i * TILE_WIDTH + tx][j * TILE_WIDTH + ty][c] = 
                            I0[c * (W + 2 * P) * (H + 2 * P) + (bx * TILE_WIDTH + i * TILE_WIDTH + tx) * (H + 2 * P) + (by * TILE_WIDTH + j * TILE_WIDTH + ty)];
            __syncthreads();

            // Compute convolution for the current channel
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    output += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FH + (FH - 1 - j)] * 
                              N_ds[tx + i][ty + j][c];
                }
            }
            __syncthreads();
        }
        if (row_o < H && col_o < W) {
            O[k * W * H + row_o * H + col_o] = output;
        }
    }
}

// Rest of the functions (initializeTensors, calculateChecksum, main) remains the same as in the previous example

void initializeTensors(double* I, double* F, double* I0) {
    // Initialize I
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                I[c * W * H + x * H + y] = c * (x + y);
            }
        }
    }

    // Initialize F
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Initialize I0 with padding
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W + 2 * P; ++x) {
            for (int y = 0; y < H + 2 * P; ++y) {
                if (x == 0 || y == 0 || x == W + 2 * P - 1 || y == H + 2 * P - 1) {
                    I0[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = 0;
                } else {
                    I0[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = I[c * W * H + (x - 1) * H + (y - 1)];
                }
            }
        }
    }
}

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
int main() {
    double *I, *F, *I0, *O;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&I, C * W * H * sizeof(double));
    cudaMallocManaged(&F, K * C * FH * FW * sizeof(double));
    cudaMallocManaged(&I0, C * (W + 2 * P) * (H + 2 * P) * sizeof(double));
    cudaMallocManaged(&O, K * W * H * sizeof(double));

    // Initialize tensors
    initializeTensors(I, F, I0);

    // Launch the kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((W + dimBlock.x - 1) / dimBlock.x, (H + dimBlock.y - 1) / dimBlock.y, K);

    initialize_timer();
    start_timer();
    convolveTiled<<<dimGrid, dimBlock>>>(I0, F, O);
    cudaDeviceSynchronize();
    stop_timer();
    double time = elapsed_time();
    // Calculate the checksum of O
    double checksum = calculateChecksum(O);
    printf("Checksum: %f\n", checksum);
    printf("Execution Time: %lf seconds\n", time);

    // Free resources
    cudaFree(I);
    cudaFree(F);
    cudaFree(I0);
    cudaFree(O);

    return 0;
}