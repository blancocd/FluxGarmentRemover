from PIL import Image
from remove_garment_mv import remove_garment_kontext
from utils.concat import transp_to_white
import os
import sys
import random
import numpy as np
import json
MAX_SEED = np.iinfo(np.int32).max

# Test the seeds and prompts that will be used for multiview generation.
# First the front view image is copied, then the outer garment is removed for Outer scans
# for Inner scans it's just copied as {scan_name}_outer.png since it doesn't have outer garment
# then the inner garment is removed and the generated image is saved as {scan_name}_inner.png
# lastly, the lower garment is removed and the generated image is saved as {scan_name}_lower.png
def main(dataset_dir, garment_data_json, index):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)

    scan_names = list(garment_data.keys())
    scan_name = scan_names[index-1]
    scan_dict = garment_data[scan_name]
    print(f"Processing scan {scan_name} with index {index-1}.")

    copy_filename = f"./test_fkon/{scan_name}.png"
    outer_filename = f"./test_fkon/{scan_name}_outer.png"
    if not os.path.isfile(copy_filename):
        scan_dir = os.path.join(dataset_dir, scan_name)
        initial_anchor_idx = scan_dict['anchor_idx']

        image_path = os.path.join(scan_dir, 'images', f'train_{initial_anchor_idx:04d}.png')
        scan_image = transp_to_white(Image.open(image_path))
        scan_image.save(copy_filename)

        if 'outer' not in scan_dict:
            scan_image.save(outer_filename)
        
    if not os.path.isfile(outer_filename):
        scan_image = Image.open(copy_filename)
        prompt = scan_dict['outer']['prompt']
        seed = scan_dict['outer']['seed']
        seed = random.randint(0, MAX_SEED) if seed == -1 else seed
        print(f'Will remove outer garment for {scan_name} with prompt {prompt} and seed {seed}.')
        gen_image = remove_garment_kontext(scan_image, prompt, seed=seed)
        outer_filename = f"./test_fkon/{scan_name}_outer_{seed}.png"
        gen_image.save(outer_filename)
        print(f"Generated image saved as {outer_filename}")
        
    # inner_filename = f"./test_fkon/{scan_name}_inner.png"
    # if not os.path.isfile(inner_filename):
    #     image_no_outer = Image.open(outer_filename)
    #     prompt = scan_dict['inner']['prompt']
    #     seed = scan_dict['inner']['seed']
    #     seed = random.randint(0, MAX_SEED) if seed == -1 else seed
    #     inner_filename = f"./test_fkon/{scan_name}_inner_{seed}.png"
    #     print(f'Will remove inner garment for {scan_name} with prompt {prompt} and seed {seed}.')
    #     gen_image = remove_garment_kontext(image_no_outer, prompt, seed=seed)
    #     gen_image.save(inner_filename)
    #     print(f"Generated image saved as {inner_filename}")

    # lower_filename = f"./test_fkon/{scan_name}_lower.png"
    # if not os.path.isfile(lower_filename):
    #     prompt = scan_dict['lower']['prompt']
    #     seed = scan_dict['lower']['seed']
    #     seed = random.randint(0, MAX_SEED) if seed == -1 else seed
    #     print(f'Will remove lower garment for {scan_name} with prompt {prompt} and seed {seed}.')
    #     gen_image = remove_garment_kontext(image, prompt, seed=seed)
    #     gen_image.save(lower_filename)
    #     print(f"Generated image saved as {lower_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <dataset_dir> <garment_data_json> <index>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    garment_data_json = sys.argv[2]
    index = int(sys.argv[3])
    main(dataset_dir, garment_data_json, index)
