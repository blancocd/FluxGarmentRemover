import os
import gc

from utils.concat import concatenate_imgs, transp_to_white
from utils.create_masks_from_seg import get_mask_4ddress
from utils.deconcat import deconcat_img, save_new_segmap
import logging
from PIL import Image
from huggingface_hub import login
import numpy as np
from diffusers import FluxKontextPipeline, FluxFillPipeline
import torch

# token = os.getenv("HUGGINGFACE_TOKEN")
# login(token=token)

import random
MAX_SEED = np.iinfo(np.int32).max

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

def remove_garment_kontext(image, prompt, seed = None):
    pipe_kontext = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16, safety_checker=None).to("cuda")
    pipe_kontext.safety_checker = disabled_safety_checker
    h, w = (image.height, image.width) if isinstance(image, Image.Image) else (image.shape[0], image.shape[1])
    seed =  seed or random.randint(0, MAX_SEED)
    print(f'Flux Kontext seed is {seed}')
    gen_image = pipe_kontext(
        image=image,
        prompt=prompt,
        height=h,
        width=w,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    
    del pipe_kontext
    torch.cuda.empty_cache()
    return gen_image

def remove_garment_fill(pipe, image, mask, prompt, seed = None):
    h, w = (image.height, image.width) if isinstance(image, Image.Image) else (image.shape[0], image.shape[1])
    seed =  seed or random.randint(0, MAX_SEED)
    print(f'Flux Fill seed is {seed}')
    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    
    torch.cuda.empty_cache()
    return gen_image

def get_equally_spaced_anchors_indices(initial_anchor_idx, num_views, num_anchors):
    anchor_indices = [(initial_anchor_idx + (i * round(num_views / num_anchors))) % num_views for i in range(num_anchors)]
    
    anchor_groups = {i: [] for i in anchor_indices}
    for index_to_assign in range(num_views):
        min_distance = float('inf')
        closest_anchor = -1
        for anchor in anchor_indices:
            direct_dist = abs(index_to_assign - anchor)
            wrap_dist = num_views - direct_dist
            distance = min(direct_dist, wrap_dist)
            if distance < min_distance:
                min_distance = distance
                closest_anchor = anchor
        anchor_groups[closest_anchor].append(index_to_assign)

    indices_list, indices_to_gen_save_flag_list = [anchor_indices], [[False] + ([True] * (len(anchor_indices)-1))]
    for anchor_idx in anchor_indices:
        indices = anchor_groups[anchor_idx]
        indices_to_gen_save = [i!=anchor_idx for i in indices]
        if any(indices_to_gen_save):
            indices_list.append(indices)
            indices_to_gen_save_flag_list.append(indices_to_gen_save)

    return indices_list, indices_to_gen_save_flag_list
    

def get_sweeping_anchors_indices(initial_anchor_idx, num_views):
    curr_anchors = [initial_anchor_idx, initial_anchor_idx]
    completed_indices = [initial_anchor_idx]
    indices_list, indices_to_gen_save_flag_list = [], []
    while len(completed_indices) != num_views:
        indices = list(range(curr_anchors[0]-2, curr_anchors[0]+1)) + list(range(curr_anchors[1], curr_anchors[1]+3))
        indices = list(dict.fromkeys([(i + num_views) % num_views for i in indices]))
        curr_anchors = [indices[0], indices[-1]]
        indices_to_gen_save = [i not in completed_indices for i in indices]
        completed_indices.extend([i for i in indices if i not in completed_indices])
        indices_list.append(indices)
        indices_to_gen_save_flag_list.append(indices_to_gen_save)
        
    sorted_pairs = sorted(zip(indices_list[-1], indices_to_gen_save_flag_list[-1]), key=lambda pair: pair[0], reverse=True)
    indices_list[-1], indices_to_gen_save_flag_list[-1] = zip(*sorted_pairs)
    return indices_list, indices_to_gen_save_flag_list


def remove_garment_anchors(scan_dir, garment_type, prompt_flux_kontext, prompt_flux_fill, 
                           initial_anchor_idx, indices_list, indices_to_gen_save_flag_list,
                           seed_flux_kontext=None, seed_flux_fill=None, 
                           ratio=4, pixel_sep=20, dil_its=1, ero_its=1, verbose = False):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    def vcomment(msg):
        if verbose:
            logger.info(msg)

    # Preparing sets of images that will be concatenated together
    img_dir = os.path.join(scan_dir, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])

    # Remove garment from view with Flux Kontext. White bg works better with these models.
    front_view_img = transp_to_white(Image.open(os.path.join(scan_dir, 'images', img_fns[initial_anchor_idx])))
    gen_front_view_img = remove_garment_kontext(front_view_img, prompt_flux_kontext, seed=seed_flux_kontext)
    gen_front_view_img.save(os.path.join(scan_dir, 'images', f'gen_{initial_anchor_idx:04d}.png'))
    save_new_segmap(scan_dir, initial_anchor_idx)
    vcomment(f'Removed garment from front view image: {initial_anchor_idx} and saved it.')
    del gen_front_view_img; gc.collect(); torch.cuda.empty_cache()

    # Load FluxFill pipeline
    pipe_fill = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, safety_checker=None).to("cuda")
    pipe_fill.safety_checker = disabled_safety_checker

    vcomment(f"Starting {len(indices_list)} iterations to generate all views:")
    for indices, indices_to_gen_save_flag in zip(indices_list, indices_to_gen_save_flag_list):
        anchor_indices = [i for i, f in zip(indices, indices_to_gen_save_flag) if not f]
        indices_to_gen_save = [i for i, f in zip(indices, indices_to_gen_save_flag) if f]
        vcomment(f"Anchor indices {anchor_indices} will be used to generate {indices_to_gen_save}.")
        
        # Loading images based on whether they are without garment and thus anchor or to be generated
        concat_imgs, concat_segs = [], []
        for i, gen_save, in zip(indices, indices_to_gen_save_flag):
            if gen_save:
                img_fn = os.path.join(scan_dir, 'images', f'train_{i:04d}.png')
                concat_imgs.append(Image.open(img_fn))
                seg_fn = os.path.join(scan_dir, 'segmentation_masks', f'train_{i:04d}.png')
                concat_segs.append(Image.open(seg_fn))
            else:
                img_fn = os.path.join(scan_dir, 'images', f'gen_{i:04d}.png')
                concat_imgs.append(Image.open(img_fn))
                concat_segs.append(None)

        # Concatenate images and segmentation maps
        concat_img, concat_seg, concat_img_coords_list, human_dims_list = concatenate_imgs(
            concat_imgs, concat_segs, ratio=ratio, pixel_sep=pixel_sep)
        vcomment("Concatenated views.")

        # Get mask of concatenated anchor images according to type(s) of inpainting
        mask = get_mask_4ddress(concat_seg, garment_type, dil_its=dil_its, ero_its=ero_its).astype(np.float32)

        vcomment(f"Mask of type(s): {garment_type} is ready for concatenated views.")

        # Inpaint concatentated anchor images with FluxFill:
        concat_img_pil = Image.fromarray(concat_img.astype(np.uint8))
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        gen_concat_images = remove_garment_fill(pipe_fill, concat_img_pil, mask_pil, prompt_flux_fill, 
                                                seed=seed_flux_fill)
        vcomment(f"Removed garments from concatenated views.")
        
        # Deconcatenate images and save generated images
        deconcat_img(scan_dir, gen_concat_images, indices, concat_img_coords_list, human_dims_list, 
                                  indices_to_gen_save_flag=indices_to_gen_save_flag)
        vcomment(f"{indices_to_gen_save} have been saved to scan directory.")
        del gen_concat_images; gc.collect(); torch.cuda.empty_cache()


def get_initial_anchor_idx(scan_dir, img_fns):
    count_pixels = []
    for img_fn in img_fns:
        seg_path = os.path.join(scan_dir, 'segmentation_masks', img_fn)
        seg_map = np.array(Image.open(seg_path).convert('RGB'))
        human_mask = get_mask_4ddress(seg_map, 'human', dil_its=0, ero_its=None)
        skin_mask = get_mask_4ddress(seg_map, 'skin', dil_its=0, ero_its=None)
        inner_mask = get_mask_4ddress(seg_map, 'inner', dil_its=0, ero_its=None)
        count_pixels.append(human_mask.sum()+2*skin_mask.sum()+inner_mask.sum())
    highest_count = max(count_pixels)
    return count_pixels.index(highest_count)

'''
garment_type = 'upper'
prompt_flux_kontext = 'remove the outer garment'
prompt_flux_fill = 'white long sleeve shirt'
seed_flux_kontext = 0
seed_flux_fill = 0

img_dir = os.path.join(scan_dir, 'images')
img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])
num_views = len(img_fns)
initial_anchor_idx = get_initial_anchor_idx(scan_dir, img_fns)

indices_list, indices_to_gen_save_flag_list = get_sweeping_anchors_indices(initial_anchor_idx, num_views)

# num_anchors = 4
# indices_list, indices_to_gen_save_flag_list = get_equally_spaced_anchors_indices(initial_anchor_idx, num_views, num_anchors)
remove_garment_anchors(scan_dir, garment_type, prompt_flux_kontext, prompt_flux_fill, 
                       initial_anchor_idx, indices_list, indices_to_gen_save_flag_list,
                       seed_flux_kontext=seed_flux_kontext, seed_flux_fill=seed_flux_fill, verbose = True)
'''