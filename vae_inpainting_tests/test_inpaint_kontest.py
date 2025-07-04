import torch 
from diffusers import FluxKontextInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
from 

pipe = FluxKontextInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
prompt = ""

input_image = load_image('/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/00122_Outer/images/train_0000.png')
mask_image = Image.new("RGB", (input_image.width, input_image.height))

image = pipe(prompt=prompt, image=input_image, mask_image=mask_image).images[0]
image.save("kontext_inpainting_ref.png")
