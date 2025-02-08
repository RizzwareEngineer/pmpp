__global__ void col_matmul_kernel(float* M, float* N, float* P, int rows, int K, int cols) {
    int thread_col = blockDim.y * blockIdx*y + threadIdx.y;

    for (int curr_row = 0; curr_row < rows; ++curr_row) {
        
    }
}