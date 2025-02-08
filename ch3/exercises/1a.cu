#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void row_matmul_kernel(const float *M, const float *N, float *P, int rows, int K, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(row < rows))
        return;

    for (int col = 0; col < cols; ++col)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            {
                sum += M[row * K + k] * N[k * cols + col];
            }
        P[row * cols + col] = sum;
    }
}

void row_matmul(
    torch::Tensor M,
    torch::Tensor N,
    torch::Tensor P,
    int rows,
    int K,
    int cols
)
{
    // Ensure our Tensors are on GPU
    TORCH_CHECK(M.is_cuda(), " M must be a CUDA tensor");



    int blockSize = 256; // As long as 32-multiple
    int gridSize = (rows + blockSize - 1) / blockSize;

    row_matmul_kernel<<<gridSize, blockSize>>>(
        M.data_ptr<float>(), 
        N.data_ptr<float>(), 
        P.data_ptr<float>(), 
        rows, 
        K, 
        cols
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_matmul", &row_matmul, "Row-based MatMul");
}