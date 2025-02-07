from torch.utils.cpp_extension import load
import torchvision.transforms as T
from PIL import Image

# Build kernel extension
blur = load(
    name="blur",
    sources=["kernel.cu"],
    verbose=True
)

def blur_image(img_path):
    # Load image in grayscale
    # Why? Because this CUDA kernel is designed to work with 1-channel images
    img = Image.open(img_path).convert("L")

    # Convert to a PyTorch tensor on CUDA
    # shape => [1, H, W], values in [0..255] (uint8)
    # TODO: research
    img_tensor = T.ToTensor()(img).mul(255).byte().cuda()

    # Call it
    blurred_tensor = blur.blur(img_tensor)

    # Convert back to PIL image for viewing
    blurred_tensor = blurred_tensor.squeeze(0).cpu()  # shape => [H, W]
    blurred_image = T.ToPILImage()(blurred_tensor)

    blurred_image.save('output.png')

if __name__ == "__main__":
    blur_image("input.png")