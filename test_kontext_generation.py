from PIL import Image
from remove_garment_mv import remove_garment_kontext
import random
import numpy as np
MAX_SEED = np.iinfo(np.int32).max

# Load the image and prompt once, outside the loop
initial_anchor_idx = 0
image = Image.open('my_scan_dir/images/train_{initial_anchor_idx:04d}.png')
prompt = 'remove the outer garment for a polo shirt'
seed = random.randint(0, MAX_SEED)
gen_image = remove_garment_kontext(image, prompt, seed=seed)
gen_image.save(f"test_{seed}.png")
