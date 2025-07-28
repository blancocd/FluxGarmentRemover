from PIL import Image
from remove_garment_mv import get_initial_anchor_idx, remove_garment_kontext
from utils.concat import transp_to_white
import os
import sys
import random
import numpy as np
MAX_SEED = np.iinfo(np.int32).max

def main(dataset_dir, index):
    scan_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and 
                        'Outer' in d])
    
    scan_dir = os.path.join(dataset_dir, scan_names[index])
    img_dir = os.path.join(scan_dir, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])
    initial_anchor_idx = get_initial_anchor_idx(scan_dir, img_fns)

    image_path = os.path.join(scan_dir, 'images', f'train_{initial_anchor_idx:04d}.png')
    image = transp_to_white(Image.open(image_path))
    
    prompt = 'remove the outer garment'
    seed = random.randint(0, MAX_SEED)
    gen_image = remove_garment_kontext(image, prompt, seed=seed)
    output_filename = f"./flux_test_imgs/{scan_names[index]}_{seed}.png"
    gen_image.save(output_filename)
    print(f"Generated image saved as {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: test_kontext_generation.py dataset_dir <index>")
        sys.exit(1)
    
    index = int(sys.argv[2])
    dataset_dir = sys.argv[1]
    main(dataset_dir, index)
