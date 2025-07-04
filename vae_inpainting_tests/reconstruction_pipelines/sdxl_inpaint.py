# pipeline_stable_diffusion_xl_reconstruction.py

import inspect
import torch
from PIL import Image
import logging
from typing import Optional, Union, List, Dict, Any

# This pipeline inherits from StableDiffusionXLInpaintPipeline.
# Ensure you have `diffusers` installed.
from diffusers import StableDiffusionXLInpaintPipeline

# It's good practice to have a logger configured.
# If you run this in a project without logging setup, messages might not appear.
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transp_to_white(image):
    if image.mode != 'RGBA':
        return image

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image = new_image.convert("RGB")
    return new_image

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionXLReconstructionPipeline(StableDiffusionXLInpaintPipeline):
    """
    A modified pipeline to test the reconstruction capabilities of the Stable Diffusion XL Inpainting model.
    This pipeline runs the full inpainting denoising process on an unmasked image with
    an empty prompt to measure how well the original image is preserved.

    It returns the final denoised image, a baseline VAE reconstruction, and optional
    metrics comparing the original and denoised latents.
    """

    def _compute_metrics(self, original_latents: torch.Tensor, denoised_latents: torch.Tensor) -> Dict[str, float]:
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
            data_range = original_latents.max() - original_latents.min()
            if data_range == 0:
                metrics["psnr"] = float('inf')
            else:
                psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
                metrics["psnr"] = psnr.item()

        return metrics

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0, # Guidance is irrelevant with empty prompt
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_metrics: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Runs the reconstruction experiment for the SDXL inpainting pipeline.

        Args:
            image (`Optional[torch.FloatTensor]`): The input image. If None, a random RGB image is generated.
            height, width (`int`): Dimensions for the image.
            strength (`float`): The strength of the noise added. 1.0 means full noise.
            num_inference_steps (`int`): The number of steps for the denoising loop.
            guidance_scale (`float`): Set to 0.0 as it has no effect with an empty prompt.
            generator (`torch.Generator`): Torch generator for reproducibility.
            return_metrics (`bool`): If True, computes and returns MSE and PSNR on the latents.

        Returns:
            A dictionary containing: `denoised_image`, `reconstructed_image`, and optional `metrics`.
        """
        # Get defaults from original pipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Internally set fixed parameters for the experiment
        prompt = ""
        negative_prompt = ""
        batch_size = 1
        num_images_per_prompt = 1
        device = self._execution_device
        
        # 1. Prepare Inputs (Image, Mask)
        if image is None:
            logger.info(f"No image provided, creating a random RGB image of size {height}x{width}.")
            image = torch.rand((batch_size, 3, height, width), device=device, generator=generator)

        # Create a black mask (all zeros)
        mask_image = torch.zeros((batch_size, height, width), dtype=torch.uint8)
        mask_image = Image.fromarray(mask_image.squeeze().cpu().numpy(), mode="L")
        
        # Preprocess image (using the base pipeline's methods)
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32, device=device)

        # 2. Get Baseline VAE Reconstruction
        clean_original_latents = self._encode_vae_image(image=init_image, generator=generator)
        reconstructed_image_latents = clean_original_latents / self.vae.config.scaling_factor
        reconstructed_image_decoded = self.vae.decode(reconstructed_image_latents, return_dict=False)[0]
        reconstructed_image_pil = self.image_processor.postprocess(reconstructed_image_decoded, output_type="pil")[0]

        # 3. Run the full diffusion pipeline
        # Encode prompt (will be empty)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False, negative_prompt=negative_prompt
        )
        
        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self, num_inference_steps, device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        
        # Prepare latents (add noise to original image)
        denoising_latents, noise, image_latents = self.prepare_latents(
            batch_size, self.unet.config.in_channels, height, width, prompt_embeds.dtype, device, generator,
            image=init_image, timestep=latent_timestep, is_strength_max=(strength==1.0), return_noise=True, return_image_latents=True
        )

        # Prepare mask latents (using the black mask)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_image, init_image, batch_size, height, width, prompt_embeds.dtype, device, generator, do_classifier_free_guidance=False
        )
        
        # Prepare added time ids
        add_time_ids, _ = self._get_add_time_ids(
            (height, width), (0,0), (height, width), aesthetic_score=6.0, negative_aesthetic_score=2.5,
            negative_original_size=(height, width), negative_crops_coords_top_left=(0,0), negative_target_size=(height, width),
            dtype=prompt_embeds.dtype
        )
        add_time_ids = add_time_ids.to(device)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(denoising_latents, t)
                
                # The UNet for SDXL inpainting can have 9 input channels
                if self.unet.config.in_channels == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                    return_dict=False
                )[0]

                denoising_latents = self.scheduler.step(noise_pred, t, denoising_latents, return_dict=False)[0]
                
                # If UNet has 4 channels, it uses latent blending
                if self.unet.config.in_channels == 4:
                    init_latents_proper = image_latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(init_latents_proper, noise, torch.tensor([noise_timestep]))
                    
                    # With a black mask (init_mask=0), this becomes: latents = 1 * init_latents_proper + 0 * latents
                    denoising_latents = (1 - mask) * init_latents_proper + mask * denoising_latents

                progress_bar.update()

        # Decode final image
        final_denoised_latents = denoising_latents
        final_denoised_latents_for_decode = final_denoised_latents / self.vae.config.scaling_factor
        denoised_image_decoded = self.vae.decode(final_denoised_latents_for_decode, return_dict=False)[0]
        denoised_image_pil = self.image_processor.postprocess(denoised_image_decoded, output_type="pil")[0]
        
        # 4. Compute Metrics
        output_dict = {
            "denoised_image": denoised_image_pil,
            "reconstructed_image": reconstructed_image_pil
        }

        if return_metrics:
            metrics = self._compute_metrics(clean_original_latents, final_denoised_latents)
            output_dict["metrics"] = metrics

        return output_dict


# Example of how to use this new pipeline:
if __name__ == '__main__':
    from diffusers.utils import load_image

    print("Running StableDiffusionXLReconstructionPipeline example...")
    
    try:
        pipe = StableDiffusionXLReconstructionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Could not load pipeline. Error: {e}")
        exit()

    print("Running reconstruction experiment...")

    import numpy as np
    import random
    MAX_SEED = np.iinfo(np.int32).max
    input_image = transp_to_white(Image.open('/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/00122_Outer/images/train_0000.png'))
    for i in range(3):
        seed = random.randint(0, MAX_SEED)
        output = pipe(
            image=input_image,
            strength=1.0, # Use full strength to see the maximum effect of the diffusion process
            num_inference_steps=50,
            return_metrics=True, # Ask for the latent comparison
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        output["denoised_image"].save("sdxl_reconstruction_denoised_{seed}.png")
        output["reconstructed_image"].save("sdxl_reconstruction_vae_only_{seed}.png")
        print("Saved 'sdxl_reconstruction_denoised.png' and 'sdxl_reconstruction_vae_only.png' with seed {seed}")

        if "metrics" in output:
            print("\n--- Latent Comparison Metrics (SDXL) ---")
            print(f"  Mean Squared Error (MSE): {output['metrics']['mse']:.6f}")
            print(f"  Peak Signal-to-Noise Ratio (PSNR): {output['metrics']['psnr']:.2f} dB")
            print("----------------------------------------")
