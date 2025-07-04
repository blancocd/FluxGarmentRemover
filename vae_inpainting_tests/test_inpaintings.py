from reconstruction_pipelines.flux_inpaint_kontext import FluxKontextInpaintPipeline
from PIL import Image
import torch

def transp_to_white(image):
    if image.mode != 'RGBA':
        return image

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image = new_image.convert("RGB")
    return new_image

def compute_metrics(original_latents: torch.Tensor, denoised_latents: torch.Tensor):
    """Helper to compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
    between two latent tensors."""
    metrics = {}
    # Ensure tensors are float for calculations
    original_latents = original_latents.to(torch.float32)
    denoised_latents = denoised_latents.to(torch.float32)

    mse = torch.mean((original_latents - denoised_latents) ** 2)
    metrics["mse"] = mse.item()

    if mse == 0:
        metrics["psnr"] = float('inf')
    else:
        # Calculate PSNR. We use the data range of the original latents as the max value.
        data_range = original_latents.max() - original_latents.min()
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        metrics["psnr"] = psnr.item()
    print(metrics['psnr'])
    return metrics

# Example of how to use this new pipeline:
if __name__ == '__main__':
    pipe = FluxKontextInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Running reconstruction experiment...")
    import numpy as np
    import random
    MAX_SEED = np.iinfo(np.int32).max
    input_image = transp_to_white(Image.open('/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/00122_Outer/images/train_0000.png'))
    mask_image = Image.new("L", (input_image.width, input_image.height), 0) # 0 for black
    for i in range(3):
        seed = random.randint(0, MAX_SEED)
        output = pipe(
            image=input_image,
            mask_image=mask_image,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        output["denoised_image"].save("fill_reconstruction_denoised_{seed}.png")
        output["reconstructed_image"].save("fill_reconstruction_vae_only_{seed}.png")
        print("Saved 'fill_reconstruction_denoised.png' and 'fill_reconstruction_vae_only.png' with seed {seed}")

        compute_metrics(output['encoded_latents'], output['denoised_latents'])
