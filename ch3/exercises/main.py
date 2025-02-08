import torch
from torch.utils.cpp_extension import load

module = load(
    name='row_matmul',
    sources=['1a.cu'],
    verbose=True
)

def row_matmul(M: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    rows, MK = M.shape
    NK, cols = N.shape

    assert MK == NK, "Inner dimensions mismatch."
    K = MK = NK

    P = torch.zeros((rows, cols), device=M.device, dtype=M.dtype)

    module.row_matmul(M, N, P, rows, K, cols)
    return P

def main():
    # Create some CUDA tensors
    rows, K, cols = 2, 3, 4
    M = torch.arange(1, rows * K + 1, device='cuda', dtype=torch.float32).reshape(rows, K)
    N = torch.arange(1, K * cols + 1, device='cuda', dtype=torch.float32).reshape(K, cols)

    # Our custom kernel
    P_gpu = row_matmul(M, N)

    # PyTorch's built-in matmul for reference
    P_torch = M @ N

    assert torch.equal(P_gpu, P_torch), "Your kernel is incorrect."
    print('Your kernel is correct.')

if __name__ == "__main__":
    main()