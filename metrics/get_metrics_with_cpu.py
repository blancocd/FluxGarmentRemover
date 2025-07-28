import json
import os
from skimage.metrics import structural_similarity
import numpy as np
from ..utils.create_masks_from_seg import get_mask_4ddress, get_mask_from_segmap, fourddress_palette
from PIL import Image
import numpy as np
import sys

def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def get_masked_ssim(img_1_fn, img_2_fn, masks):
    _, ssim_map = structural_similarity(np.array(Image.open(img_1_fn).convert('L')), 
                                        np.array(Image.open(img_2_fn).convert('L')), full=True)
    to_return = []
    for mask in masks:
        to_return.append(np.mean(ssim_map[mask]))
    return to_return

def get_masked_psnr(img_1_fn, img_2_fn, masks):
    img_1_arr = np.array(Image.open(img_1_fn)).astype(np.float32) / 255.
    img_2_arr = np.array(Image.open(img_2_fn)).astype(np.float32) / 255.
    squared_diff_map = (img_1_arr - img_2_arr) ** 2
    to_return = []
    for mask in masks:
        mse_masked = np.mean(squared_diff_map[mask])
        psnr_masked = 10 * np.log10(1. / mse_masked)
        to_return.append(psnr_masked)
    return to_return

def get_ious(segformer_fn, gen_segformer_fn):
    segformer_map = np.array(Image.open(segformer_fn).convert('RGB'))
    gen_segformer_map = np.array(Image.open(gen_segformer_fn).convert('RGB'))
    ious = []
    for target_color in fourddress_palette:
        segformer_mask = get_mask_from_segmap(segformer_map, target_color, [], dil_its=0, ero_its = 0)
        gen_segformer_mask = get_mask_from_segmap(gen_segformer_map, target_color, [], dil_its=0, ero_its = 0)
        ious.append(iou(segformer_mask, gen_segformer_mask))
    return ious
    

def main(method, dataset_dir, gen_dir, dil_its=1, ero_its=1):
    gen_dir = os.path.join(gen_dir, method)

    scan_names = sorted([d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))])
    results_dict = {}
    for scan_name in scan_names:
        gen_paths = sorted([f for f in os.listdir(os.path.join(gen_dir, scan_name, 'images')) if f.startswith('gen')])
        results_dict[scan_name] = {
            'indices': [],
            'ssim_inner_mask': [],
            'ssim_nongen_mask': [],
            'psnr_inner_mask': [],
            'psnr_nongen_mask': [],
            'ious': []
        }
        for gen_path in gen_paths:
            idx = int(gen_path.split('_')[1].split('.')[0])
            img_fn = os.path.join(dataset_dir, scan_name, 'images', gen_path)
            seg_fn = os.path.join(dataset_dir, scan_name, 'segmentation_masks', gen_path)
            gen_img_fn = os.path.join(gen_dir, scan_name, 'images', gen_path)
            
            seg_map = np.array(Image.open(seg_fn).convert('RGB'))
            inner_mask = get_mask_4ddress(seg_map, 'inner', dil_its=0, ero_its=None)
            human_mask = get_mask_4ddress(seg_map, 'human', dil_its=0, ero_its=None)
            # The dil_its and ero_its must match those used during generation
            outer_mask = get_mask_4ddress(seg_map, 'outer', dil_its=dil_its, ero_its=ero_its)
            shouldnt_have_edited_mask = np.logical_and(human_mask, ~outer_mask)
            
            ssim_inner_mask, ssim_nongen_mask = get_masked_ssim(img_fn, gen_img_fn, [inner_mask, shouldnt_have_edited_mask])
            psnr_inner_mask, psnr_nongen_mask = get_masked_psnr(img_fn, gen_img_fn, [inner_mask, shouldnt_have_edited_mask])

            segformer_fn = os.path.join(dataset_dir, scan_name, 'segformer_segmentation_masks', gen_path)
            gen_segformer_fn = os.path.join(gen_dir, scan_name, 'segformer_segmentation_masks', gen_path)
            ious = get_ious(segformer_fn, gen_segformer_fn)

            results_dict[scan_name]['indices'].append(idx)
            results_dict[scan_name]['ssim_inner_mask'].append(ssim_inner_mask)
            results_dict[scan_name]['ssim_nongen_mask'].append(ssim_nongen_mask)
            results_dict[scan_name]['psnr_inner_mask'].append(psnr_inner_mask)
            results_dict[scan_name]['psnr_nongen_mask'].append(psnr_nongen_mask)
            results_dict[scan_name]['ious'].append(ious)

    results_fn = print(f'{method}_cpu_results.json')
    with open(results_fn, 'w') as f:
        json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Example usage: {sys.argv[0]} <method>")
    else:
        method = sys.argv[1]
        dataset_dir = sys.argv[2]
        gen_dir = sys.argv[3]
        print(f'Using method {method}.')
        main(method, dataset_dir, gen_dir)
