// matrix_multiplication.cpp
#include <torch/extension.h>

// Declaration of the CUDA function
void add_cuda(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);
void mul_cuda(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c);

// C++ function that wraps the CUDA function
torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b) {
    int height = a.size(0);
    int width = a.size(1);
    auto c = torch::empty({height,width}, a.options());
    add_cuda(a, b, c);
    return c;
}

torch::Tensor mul(const torch::Tensor& a, const torch::Tensor& b) {
    int height = a.size(0);
    int width = a.size(1);
    auto c = torch::empty({height,width}, a.options());
    mul_cuda(a, b, c);
    return c;
}

// Pybind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors using CUDA");
    m.def("mul", &mul, "mul two tensors using CUDA");
}