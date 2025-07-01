import os
import numpy as np
from PIL import Image

fourddress_palette = np.array([
    [255., 128.,   0.],   # 0 hair
    [128., 128., 128.],   # 1 body
    [255.,   0.,   0.],   # 2 inner
    [  0., 128., 255.],   # 3 outer
    [  0., 255.,   0.],   # 4 pants
    [128.,   0., 255.]    # 5 shoes
])

def deconcat_img(scan_dir, concat_gen_img, indices, concat_img_coords_list, human_dims_list, indices_to_gen_save_flag = None):
    indices_to_gen_save_flag = indices_to_gen_save_flag or True * len(indices)

    if isinstance(concat_gen_img, Image.Image):
        concat_gen_img = np.array(concat_gen_img.convert('RGB'))

    img_dir = os.path.join(scan_dir, 'images')
    for i, save, concat_img_coords, human_dims in zip(indices, indices_to_gen_save_flag, concat_img_coords_list, human_dims_list):
        img_canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        ceiling, floor, left, right = map(int, human_dims)
        start_height, end_height, start_width, end_width = map(int, concat_img_coords)
        img_canvas[ceiling:floor, left:right] = concat_gen_img[start_height:end_height, start_width:end_width]

        if save:
            img = Image.fromarray(img_canvas)
            img.save(os.path.join(img_dir, f'gen_{i:04d}.png'))

    seg_dir = os.path.join(scan_dir, 'segmentation_masks')
    inner_color = fourddress_palette[2]
    outer_color = fourddress_palette[3]
    tolerance = 5
    for i, save in zip(indices, indices_to_gen_save_flag):
        seg = np.array(Image.open(os.path.join(seg_dir, f'train_{i:04d}.png')).convert('RGB'))
        close = np.all(np.abs(seg - outer_color) <= tolerance, axis=-1)
        seg[close] = inner_color

        if save:
            seg = Image.fromarray(seg)
            seg.save(os.path.join(seg_dir, f'gen_{i:04d}.png'))

def save_new_segmap(scan_dir, idx):
    seg_dir = os.path.join(scan_dir, 'segmentation_masks')
    inner_color = fourddress_palette[2]
    outer_color = fourddress_palette[3]
    tolerance = 5

    seg = np.array(Image.open(os.path.join(seg_dir, f'train_{idx:04d}.png')).convert('RGB'))
    close = np.all(np.abs(seg - outer_color) <= tolerance, axis=-1)
    seg[close] = inner_color

    seg = Image.fromarray(seg)
    seg.save(os.path.join(seg_dir, f'gen_{idx:04d}.png'))