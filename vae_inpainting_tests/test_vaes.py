import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.concat import transp_to_white
import gc

# Import all necessary pipeline and model classes
from diffusers import (
    AutoPipelineForInpainting,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
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

def main():
    # --- Configuration ---
    image_fn = "/mnt/lustre/work/ponsmoll/pba870/shared/00122_Outer/images/train_0000.png" # <--- IMPORTANT: SET YOUR IMAGE PATH HERE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # # ==================================================================
    # # 🔬 1. SDXL Inpainting
    # # ==================================================================
    # print("\n--- Evaluating VAE for: SDXL Inpainting ---")
    # try:
        # dtype = torch.float16
        # pipe_sdxl = AutoPipelineForInpainting.from_pretrained(
        #     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        #     torch_dtype=dtype,
        #     variant="fp16"
        # ).to(device)

    #     psnr, ssim = get_reconstruction_metrics('sdxl1', image_fn, pipe_sdxl, 1024, device, dtype)
    #     print(f"✅ Results: PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")

    # except Exception as e:
    #     print(f"❌ An error occurred: {e}")
    # finally:
    #     if 'pipe_sdxl' in locals(): del pipe_sdxl
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print("Cleaned up memory.")

    # # ==================================================================
    # # 🔬 3. FLUX.1-Kontext
    # # ==================================================================
    # print("\n--- Evaluating VAE for: FLUX.1-Kontext ---")
    # pipe_kontext = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
    # psnr, ssim = get_reconstruction_metrics('fluxkontext', image_fn, pipe_kontext, 1024, device, torch.bfloat16)
    # print(f"✅ Results: PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")
    # gc.collect()
    # torch.cuda.empty_cache()
    # print("Cleaned up memory.")

    # ==================================================================
    # 🔬 4. FLUX.1-Fill
    # ==================================================================
    print("\n--- Evaluating VAE for: FLUX.1-Fill ---")
    pipe_fill = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
    psnr, ssim = get_reconstruction_metrics('fluxfill', image_fn, pipe_fill, 1024, device, torch.bfloat16)
    print(f"✅ Results: PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleaned up memory.")


if __name__ == "__main__":
    main()