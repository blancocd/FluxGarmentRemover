# pipeline_flux_reconstruction.py
import inspect
import torch
from PIL import Image
import logging
from typing import Optional, Union, List, Dict, Any
from diffusers import FluxFillPipeline

# It's good practice to have a logger configured.
# If you run this in a project without logging setup, messages might not appear.
# You can uncomment the following lines for basic logging to the console.
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


class FluxReconstructionPipeline(FluxFillPipeline):
    """
    A modified pipeline to test the reconstruction capabilities of the FLUX model.
    This pipeline runs the full denoising process on an unmasked image with an empty prompt
    to measure how well the original image is preserved.

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
            # Calculate PSNR. We use the data range of the original latents as the max value.
            data_range = original_latents.max() - original_latents.min()
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
    ) -> Dict[str, Any]:
        """
        Runs the reconstruction experiment.

        Args:
            image (`Optional[torch.FloatTensor]`):
                The input image. If None, a random RGB image will be generated.
            height, width (`int`):
                Dimensions for the image if a random one is generated.
            strength (`float`):
                The strength of the noise added. 1.0 means full noise.
            num_inference_steps (`int`):
                The number of steps for the denoising loop.
            guidance_scale (`float`):
                Set to 0.0 as it has no effect with an empty prompt.
            generator (`torch.Generator`):
                Torch generator for reproducibility.
            return_metrics (`bool`):
                If True, computes and returns MSE and PSNR on the latents.

        Returns:
            A dictionary containing:
            - `denoised_image`: The final image after the full diffusion process.
            - `reconstructed_image`: The image after only VAE encoding and decoding.
            - `metrics`: A dictionary with 'mse' and 'psnr' if `return_metrics` is True.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device
        num_images_per_prompt = 1
        batch_size = 1

        # 1. Prepare Inputs (Image, Mask, Prompt)
        if image is None:
            logger.info(f"No image provided, creating a random RGB image of size {height}x{width}.")
            image = torch.rand((1, 3, height, width), device=device, generator=generator)

        # Preprocess the input image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(device=device, dtype=self.vae.dtype)


        # Internally set an empty prompt and a black mask
        prompt = ""
        mask_image = torch.zeros_like(init_image[:, :1, :, :]) # Black mask (C=1)

        # 2. Get Baseline VAE Reconstruction
        clean_original_latents_unpacked = self._encode_vae_image(image=init_image, generator=generator)
        reconstructed_image_latents = (clean_original_latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        reconstructed_image_decoded = self.vae.decode(reconstructed_image_latents, return_dict=False)[0]
        reconstructed_image_pil = self.image_processor.postprocess(reconstructed_image_decoded, output_type="pil")[0]

        # 3. Run the full diffusion pipeline
        # Prepare prompt embeddings for the empty prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=num_images_per_prompt
        )

        # Prepare timesteps based on strength
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # Prepare initial noisy latents
        num_channels_latents = self.vae.config.latent_channels
        denoising_latents, latent_image_ids = self.prepare_latents(
            init_image, latent_timestep, batch_size * num_images_per_prompt,
            num_channels_latents, height, width, prompt_embeds.dtype, device, generator,
        )

        # Prepare conditioning latents (using the black mask)
        # This will create conditioning based on the full, unmasked image.
        masked_image = init_image * (1 - mask_image)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_image, masked_image, batch_size, num_channels_latents,
            num_images_per_prompt, height, width, prompt_embeds.dtype, device, generator,
        )
        conditioning_latents = torch.cat((masked_image_latents, mask), dim=-1)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(denoising_latents.shape[0]).to(denoising_latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=torch.cat((denoising_latents, conditioning_latents), dim=2),
                    timestep=timestep / 1000,
                    guidance=None, # No guidance with empty prompt
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                denoising_latents = self.scheduler.step(noise_pred, t, denoising_latents, return_dict=False)[0]
                progress_bar.update()

        # Unpack and decode the final denoised latents
        final_denoised_latents_packed = denoising_latents
        final_denoised_unpacked = self._unpack_latents(final_denoised_latents_packed, height, width, self.vae_scale_factor)
        final_denoised_unpacked_for_decode = (final_denoised_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        denoised_image_decoded = self.vae.decode(final_denoised_unpacked_for_decode, return_dict=False)[0]
        denoised_image_pil = self.image_processor.postprocess(denoised_image_decoded, output_type="pil")[0]

        # 4. Compute Metrics if requested
        output_dict = {
            "denoised_image": denoised_image_pil,
            "reconstructed_image": reconstructed_image_pil
        }

        if return_metrics:
            # We need the "packed" version of the original latents to compare with the final packed latents
            packed_original_latents = self._pack_latents(
                clean_original_latents_unpacked, batch_size, num_channels_latents, height, width
            )
            metrics = self._compute_metrics(packed_original_latents, final_denoised_latents_packed)
            output_dict["metrics"] = metrics

        return output_dict


# Example of how to use this new pipeline:
if __name__ == '__main__':
    # This block will only run if you execute this script directly,
    # e.g., `python pipeline_flux_reconstruction.py`
    # You would need to have `pipeline_flux_fill.py` in the same directory.

    from diffusers.utils import load_image

    print("Running FluxReconstructionPipeline example...")

    # Load the pretrained model into the new pipeline class
    # Make sure you have enough memory, or enable CPU offloading on the pipe.
    try:
        pipe = FluxReconstructionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Could not load pipeline. Make sure you have diffusers and transformers installed and are logged in to Hugging Face if necessary.")
        print(f"Error: {e}")
        exit()


    # Run the experiment
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

        output["denoised_image"].save("fill_reconstruction_denoised_{seed}.png")
        output["reconstructed_image"].save("fill_reconstruction_vae_only_{seed}.png")
        print("Saved 'fill_reconstruction_denoised.png' and 'fill_reconstruction_vae_only.png' with seed {seed}")

        if "metrics" in output:
            print("\n--- Latent Comparison Metrics ---")
            print(f"  Mean Squared Error (MSE): {output['metrics']['mse']:.6f}")
            print(f"  Peak Signal-to-Noise Ratio (PSNR): {output['metrics']['psnr']:.2f} dB")
            print("---------------------------------")
