// matrix_add.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
using namespace std;

__global__ void add_kernel(const float* a, const float* b, float* c, unsigned int width, unsigned int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row*width+col;
    if ((row<height) && (col<width)) {
        c[index] = a[index] + b[index];
    }
}

void add_cuda(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    unsigned int height = a.size(0);
    unsigned int width = a.size(1);
    dim3 dimBlock(32,32);
    cout << width << height << dimBlock.x << dimBlock.y << ceil(width/dimBlock.x) << ceil(height/dimBlock.y) << endl;
    dim3 dimGrid(ceil((float)width/dimBlock.x),ceil((float)height/dimBlock.y));

    add_kernel<<<dimGrid, dimBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), width, height);
}