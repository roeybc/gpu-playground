// matrix_multiplication.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
using namespace std; 

__global__ void mul_kernel(const float* a, const float* b, float* c, unsigned int width, unsigned int height, unsigned int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row>=height || col>=width) {
        return;
    }

    float tmp = 0;
    for (int i=0; i<k; i++) {
        tmp += a[row*width+i] * b[width*i+col];
    }
    c[row*width+col] = tmp;
}

#define TILE_WIDTH 16
__global__ void mul_kernel_tiles(const float* a, const float* b, float* c, unsigned int width, unsigned int height, unsigned int k) {
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;

    float tmp = 0;
    for (int t = 0; t<ceil((float)k/TILE_WIDTH); ++t) {
        int a_id = row*k+t*TILE_WIDTH+tx;
        if (t*TILE_WIDTH+tx < k && row<height) {
            a_tile[ty][tx] = a[a_id];
        } else {
            a_tile[ty][tx] = 0;
        }

        int b_id = ty*width+t*TILE_WIDTH*width+col;
        if (ty+t*TILE_WIDTH < k && col<width) {
            b_tile[ty][tx] = b[b_id];
        } else {
            b_tile[ty][tx] = 0;
        }
        __syncthreads();

        // calclate dot product
        for (int i=0; i<TILE_WIDTH; ++i) {
            tmp += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();
    }
    if (row<height && col<width)
        c[row*width+col] = tmp;
}

void mul_cuda(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    unsigned int height = a.size(0);
    unsigned int k = a.size(1);
    unsigned int width = b.size(1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 dimGrid(ceil((float)width/dimBlock.x),ceil((float)height/dimBlock.y));

    cout << width << height << width << k << endl;
    mul_kernel_tiles<<<dimGrid, dimBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), width, height, k);
}