import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import json
import sys

def transp_to_white(image):
    if image.mode != 'RGBA':
        return image

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image = new_image.convert("RGB")
    return new_image

# Import all necessary pipeline and model classes
from diffusers import (
    AutoPipelineForInpainting,
    FluxKontextPipeline,
    FluxFillPipeline,
)

def get_reconstruction_metrics(name: str, image_fn: str, pipe, image_size: int, device: str, dtype: torch.dtype):
    """
    Loads an image, encodes and decodes it with the pipeline's VAE,
    and returns PSNR and SSIM metrics.
    """
    # 1. Load and preprocess the image
    original_pil = Image.open(image_fn).resize((image_size, image_size))
    original_pil = transp_to_white(original_pil)
    original_np = np.array(original_pil)

    image = np.array(original_pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Normalize to [-1, 1]
    image = 2.0 * image - 1.0
    image = image.to(device=device, dtype=dtype)

    vae = pipe.vae
    vae.eval()

    with torch.no_grad():
        # 2. Encode and decode
        latents = vae.encode(image).latent_dist.sample()
        reconstructed_image_tensor = vae.decode(latents).sample

    # 4. Post-process the reconstructed image
    reconstructed_image_tensor = (reconstructed_image_tensor / 2 + 0.5).clamp(0, 1)
    reconstructed_image_tensor = reconstructed_image_tensor.to(torch.float16)
    reconstructed_image_np = (reconstructed_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # 5. Calculate metrics
    psnr = peak_signal_noise_ratio(original_np, reconstructed_image_np, data_range=255)
    ssim = structural_similarity(original_np, reconstructed_image_np, channel_axis=-1, data_range=255)

    # 6. Save image
    reconstructed_pil = Image.fromarray(reconstructed_image_np)
    reconstructed_pil.save(f"vae_{name.replace(' ', '_')}.png")

    return psnr, ssim

def main(pipe_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = '/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/'
    scan_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    dtypes = [torch.float16, torch.bfloat16, torch.bfloat16]
    names = ['sdxl', 'flux_kontext', 'flux_fill']
    if pipe_idx == 0:
        pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtypes[0], variant="fp16").to(device)
    elif pipe_idx == 1:
        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=dtypes[1]).to(device)
    elif pipe_idx == 2:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=dtypes[2]).to(device)
    dtype = dtypes[pipe_idx]
    name = names[pipe_idx]

    print('using '+ name)

    results = []
    for scan_dir in scan_dirs:
        idx = np.random.randint(0, 20)
        image_fn = os.path.join(data_dir, scan_dir, 'images', f'train_{idx:04d}.png')
        psnr, ssim = get_reconstruction_metrics(name, image_fn, pipe, 1024, device, dtype)
        results.append({
            'idx': idx,
            'psnr': f"{psnr:.8f}",
            'ssim': f"{ssim:.8f}"
        })

    with open(f"vae_{name.replace(' ', '_')}_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vaes.py <pipe_idx>")
        print("Example: python test_vaes.py 0")
        sys.exit(1)
    pipe_idx = int(sys.argv[1])
    main(pipe_idx)