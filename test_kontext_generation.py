from PIL import Image
from remove_garment_mv import get_initial_anchor_idx, remove_garment_kontext
from utils.concat import transp_to_white
import os
import sys
import random
import numpy as np
import json
MAX_SEED = np.iinfo(np.int32).max

def main(dataset_dir, garment_data_json, index):
    with open(garment_data_json, 'r') as f:
        garment_data = json.load(f)

    scan_names = list(garment_data.keys())
    scan_name = scan_names[index-1]
    scan_dir = os.path.join(dataset_dir, scan_name)
    img_dir = os.path.join(scan_dir, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])
    initial_anchor_idx = get_initial_anchor_idx(scan_dir, img_fns)

    image_path = os.path.join(scan_dir, 'images', f'train_{initial_anchor_idx:04d}.png')
    image = transp_to_white(Image.open(image_path))

    copy_filename = f"./test_fkon/{scan_name}.png"
    image.save(copy_filename)
    
    prompt = garment_data[scan_name]['prompt']
    seed = garment_data[scan_name]['seed']
    seed = random.randint(0, MAX_SEED) if seed == -1 else seed
    print(f'Will generate image from base scan {scan_name} with prompt {prompt} and seed {seed}.')
    gen_image = remove_garment_kontext(image, prompt, seed=seed)
    output_filename = f"./test_fkon/{scan_name}_{seed}.png"
    gen_image.save(output_filename)
    print(f"Generated image saved as {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: test_kontext_generation.py <dataset_dir> <garment_data_json> <index>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    garment_data_json = sys.argv[2]
    index = int(sys.argv[3])
    print(index)
    main(dataset_dir, garment_data_json, index)
