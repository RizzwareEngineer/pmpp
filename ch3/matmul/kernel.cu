/*
Assumes that our matrix is square.

*/
__global__
void matmul_kernel(float* M, float* N, float* P, int w) {
    int row = blockIdx.y * blockDim.y * threadIdx.y;
    int col = blockIdx.x * blockDim.x * threadIdx.x;

    // Guard checks
    if (!(row < w) || !(col < w)) return;
    float Pvalue = 0;

    for (int i = 0; i < w; ++i) {
        Pvalue += M[row * w + i] * N[i * w + col];
    }

    P[row * w + col] = Pvalue;
}

