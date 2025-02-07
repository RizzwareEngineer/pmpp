#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

static const int BLUR_SIZE = 29;

__global__ void blur_kernel(const unsigned char *in, unsigned char *out, int w, int h)
{
    // Coordinates of current thread -> current pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure current thread/pixel is within image bounds
    if (col < w && row < h)
    {
        int pix_val = 0;
        int pixels = 0;

        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row)
        {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col)
            {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;

                // Verify we have a valid image pixel
                if (cur_row >= 0 && cur_row < h && cur_col >= 0 && cur_col < w)
                {
                    pix_val += in[cur_row * w + cur_col];
                    ++pixels;
                }
            }
        }

        // Write new pixel value
        out[row * w + col] = (unsigned char)(pix_val / pixels);
    }
}

// Wrapper function callable from Python
torch::Tensor blur(torch::Tensor input)
{
    // Ensure tensor is on CUDA and is uint8
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");

    // Input shape expected to be [1, H, W] or [H, W]; handle accordingly
    TORCH_CHECK(input.dim() == 2 || (input.dim() == 3 && input.size(0) == 1),
                "Expected shape [H, W] or [1, H, W]");

    // Extract width (W), height (H)
    int64_t H = (input.dim() == 2) ? input.size(0) : input.size(1);
    int64_t W = (input.dim() == 2) ? input.size(1) : input.size(2);

    // Create output tensor of same size on same device
    torch::Tensor output = torch::empty_like(input);

    const unsigned char *d_in = input.data_ptr<unsigned char>();
    unsigned char *d_out = output.data_ptr<unsigned char>();

    // Define block & grid
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    blur_kernel<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    return output;
}

// Expose module to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("blur", &blur, "Blur image via CUDA (uint8, single-channel)");
}