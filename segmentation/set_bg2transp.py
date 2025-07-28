import os
import sys
import torch
import numpy as np
from PIL import Image
from glob import glob

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <dataset_dir> <idx>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    idx = int(sys.argv[2])
    subject_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    scan_name = subject_names[idx]
    scan_dir = os.path.join(dataset_dir, scan_name)

    print("Procesing images from scan " + scan_name)

    img_dir = os.path.join(scan_dir, 'images')
    seg_dir = os.path.join(scan_dir, 'segmentation_masks')
    segformer_seg_dir = os.path.join(scan_dir, 'segformer_segmentation_masks')
    # Segment original render images (.startswith('train')) or generated images (.startswith('gen'))
    img_fns = sorted([f for f in os.listdir(img_dir) if f.startswith('train')]) #gen
    for img_fn in img_fns:
        img_path = os.path.join(img_dir, img_fn)
        gen_np = np.array(Image.open(img_path))[:,:,:3]

        has_orig_segmap = os.path.exists(os.path.join(seg_dir, img_fn))
        if has_orig_segmap:
            bg_mask = ~(np.array(Image.open(os.path.join(seg_dir, img_fn)))[:,:,-1].astype(bool))
        else:
            bg_mask = np.max(gen_np, axis=-1) < 2
        alpha = (np.ones(gen_np.shape[:2]) * 255).astype(np.uint8)
        alpha[bg_mask] = 0
        gen_rgba = np.dstack((gen_np, alpha))
        Image.fromarray(gen_rgba).save(img_path)

        print(f"     {os.path.basename(img_path)}")
        print(f'The new image has {(~bg_mask).sum()} non-transparent pixels')
        
        if has_orig_segmap:
            segmap = np.array(Image.open(os.path.join(seg_dir, img_fn)))[:,:,-1].astype(bool)
            print(f'The original segmentation map has {segmap.sum()} non-transparent pixels')

        segformer_segmap = np.array(Image.open(os.path.join(segformer_seg_dir, img_fn)))[:,:,-1].astype(bool)
        print(f'The segformer segmentation map has {segformer_segmap.sum()} non-transparent pixels')

if __name__ == "__main__":
    main()
