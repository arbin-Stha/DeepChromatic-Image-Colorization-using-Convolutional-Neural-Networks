from skimage import io, img_as_float
from skimage.color import gray2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Load original grayscale and colorized image
original = img_as_float(io.imread("d:/imageproject/images (1).jpg"))
colorized = img_as_float(io.imread("d:/imageproject/images (1)_1KTUlAY_colorized.jpg"))

# If the original image is grayscale, convert it to RGB to match the colorized image
if len(original.shape) == 2:  # Check if the image is grayscale
    original = gray2rgb(original)

# Compute PSNR and SSIM, with window size and data range adjustments
psnr_value = psnr(original, colorized, data_range=1.0)  # Assuming images are normalized (0 to 1)
ssim_value, _ = ssim(original, colorized, win_size=7, channel_axis=-1, full=True, data_range=1.0)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
