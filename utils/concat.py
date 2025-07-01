import os
import numpy as np
from PIL import Image
import math
from operator import itemgetter

def linear_partition(seq, k):
    if k <= 0:
        return []
    n = len(seq) - 1
    if k > n:
        return map(lambda x: [x], seq)
    table, solution = linear_partition_table(seq, k)
    k, ans = k-2, []
    while k >= 0:
        ans = [[seq[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
        n, k = solution[n-1][k], k-1
    return [[seq[i] for i in range(0, n+1)]] + ans


def linear_partition_table(seq, k):
    n = len(seq)
    table = [[0] * k for x in range(n)]
    solution = [[0] * (k-1) for x in range(n-1)]
    for i in range(n):
        table[i][0] = seq[i] + (table[i-1][0] if i else 0)
    for j in range(k):
        table[0][j] = seq[0]
    for i in range(1, n):
        for j in range(1, k):
            table[i][j], solution[i-1][j-1] = min(
                ((max(table[x][j-1], table[i][0]-table[x][0]), x) for x in range(i)),
                key=itemgetter(0))
    return (table, solution)


def get_human_height_width(img: Image, background = 1):
    img = img.convert("L")
    img_arr = np.array(img)

    if background < 127:
        img = img_arr > background 
    else:
        img = img_arr < background
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return int(rmin), int(rmax+1), int(cmin), int(cmax+1)


def crop_img(img: Image, dims: tuple | None = None, background = 1) -> np.array:
    img = img.convert("RGB")
    if dims is None:
        ceiling, floor, left, right = get_human_height_width(img, background=background)
    else:
        ceiling, floor, left, right = dims

    cropped_img = np.array(img)[ceiling:floor, left:right]

    if dims is None:
        dims = ceiling, floor, left, right
        return cropped_img, dims
    else:
        return cropped_img


# background is 0 for mask vis images and depth, 255 for images
# height pixel sep depends on whether it's the first row, middle one, or last one
# width pixel sep depends on the given width or the given one
def concat_imgs_width(imgs: list[np.ndarray], width_pixel_sep=20, height_pixel_sep_up = 20, height_pixel_sep_down = 20, width = None, background = 0) -> np.ndarray:
    heights = [img.shape[0] for img in imgs]
    widths = [img.shape[1] for img in imgs]
    channels = [img.shape[2] for img in imgs]
    assert all(i == channels[0] for i in channels)

    height = max(heights) + height_pixel_sep_up + height_pixel_sep_down
    if width is None:
        width = sum(widths) + (len(imgs) + 1) * width_pixel_sep
    
    concat_imgs_arr = np.ones((height, width, channels[0])) * background
    concat_img_coords_list = []
    for i, img in enumerate(imgs):
        # first two puts one in the vertical-center of the image and last one walks back to the beginning
        start_height = int(height_pixel_sep_up + max(heights)/2 - img.shape[0]/2)
        start_width = (i+1)*width_pixel_sep + sum(widths[:i])
        end_height = start_height + img.shape[0]
        end_width = start_width + img.shape[1]
        concat_imgs_arr[start_height: end_height, start_width: end_width] = img
        
        concat_img_coords = (start_height, end_height, start_width, end_width)
        concat_img_coords_list.append(concat_img_coords)
    
    concat_imgs_arr = concat_imgs_arr.astype(np.uint8)
    return concat_imgs_arr, concat_img_coords_list


def transp_to_white(image):
    if image.mode != 'RGBA':
        return image

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image = new_image.convert("RGB")
    return new_image


def get_per_row_flat_indices(widths, avg_height, ratio=16./9., pixel_sep=20):
    if len(widths) <= 1:
        return np.cumsum([0] + [len(widths)])
    
    ratios, partitions, widths_per_partition = [], [], []
    for n_rows in range(1, len(widths)+1):
        partition = linear_partition(widths, n_rows)
        widths_per_row = [sum(row) + (len(row)+1)*pixel_sep for row in partition]
        width = max(widths_per_row)
        height = n_rows*avg_height
        ratios.append(width/height)
        partitions.append(partition)
        widths_per_partition.append(width)
    closest_idx = np.argmin([abs(r-ratio) for r in ratios])
    closest_partition = partitions[closest_idx]
    num_entries_per_row = [len(row) for row in closest_partition]
    prefix_sum = np.cumsum([0] + num_entries_per_row)
    return prefix_sum, widths_per_partition[closest_idx]


def concat_imgs_height(imgs: list[np.ndarray], background = 0) -> np.ndarray:
    concat_img = np.concatenate(imgs, axis=0)
    new_height = int(math.ceil(concat_img.shape[0] / 16) * 16)
    new_width = int(math.ceil(concat_img.shape[1] / 16) * 16)
    concat_img_canvas = np.ones((new_height, new_width, concat_img.shape[2]), dtype=imgs[0].dtype) * background
    concat_img_canvas[:concat_img.shape[0], :concat_img.shape[1]] = concat_img
    return concat_img_canvas


def concatenate_imgs(images: list[Image.Image], segmentations: list[Image.Image | None] = None , ratio=16./9., pixel_sep=20) -> tuple[np.ndarray, np.ndarray, list, list]:
    # Crop images
    segmentations = segmentations or [None] * len(images)
    imgs, segs, human_dims_list = [], [], []
    for image, seg in zip(images, segmentations):
        if seg is None:
            cropped_img, dims = crop_img(transp_to_white(image), background=250)
            cropped_seg = np.zeros_like(cropped_img)
        else:
            cropped_seg, dims = crop_img(seg)
            cropped_img = crop_img(transp_to_white(image), dims)
        imgs.append(cropped_img)
        segs.append(cropped_seg)
        human_dims_list.append(dims)

    # Get best arrangement (number of rows and columns) based on the ratio of the image it would produce
    # This doesn't have yet the pixel separation but there should be the best arrangement with/out pixel separation
    med_height = np.median([img.shape[0] for img in imgs])
    prefix_sum, max_width = get_per_row_flat_indices([img.shape[1] for img in imgs], med_height, ratio=ratio, pixel_sep=pixel_sep)
    
    # Concatenate images along width (axis=1) separately and append them to a list of images which will be concatenated along height (axis=0)
    # Account for the height offset in the final image for all rows by offsetting their values by the previous height
    img_rows, seg_rows, concat_img_coords_list = [], [], []
    height_offset = 0
    for i, (start_idx, end_idx) in enumerate(zip(prefix_sum[:-1], prefix_sum[1:])):
        row_imgs = imgs[start_idx:end_idx]
        row_widths = [k.shape[1] for k in row_imgs]
        height_pixel_sep_up = pixel_sep if i == 0 else int(pixel_sep/2)
        height_pixel_sep_down = pixel_sep if i == (len(imgs)-1) else int(pixel_sep/2)

        width_pixel_sep = int((max_width - sum(row_widths))/(len(row_widths) + 1))
        row_concat_img, row_concat_img_coords_list = concat_imgs_width(row_imgs, width_pixel_sep=width_pixel_sep, height_pixel_sep_up=height_pixel_sep_up, 
                                                                       height_pixel_sep_down=height_pixel_sep_down, width=max_width, 
                                                                       background=255)
        row_concat_seg, _ = concat_imgs_width(segs[start_idx:end_idx], width_pixel_sep=width_pixel_sep, height_pixel_sep_up=height_pixel_sep_up, 
                                                                       height_pixel_sep_down=height_pixel_sep_down, width=max_width)

        for j, row_concat_img_coords in enumerate(row_concat_img_coords_list):
            start_height, end_height, start_width, end_width = row_concat_img_coords
            row_concat_img_coords_list[j] = (start_height + height_offset, end_height + height_offset, start_width, end_width)
        
        height_offset += row_concat_img.shape[0]
        concat_img_coords_list.extend(row_concat_img_coords_list)
        img_rows.append(row_concat_img)
        seg_rows.append(row_concat_seg)
    
    concat_img = concat_imgs_height(img_rows, background=255)
    concat_seg = concat_imgs_height(seg_rows)
    return concat_img, concat_seg, concat_img_coords_list, human_dims_list


def concat_imgs_dir(scan_dir, indices, ratio=16./9., pixel_sep=20) -> tuple[np.ndarray, np.ndarray, list, list]:
    img_dir = os.path.join(scan_dir, 'images')
    seg_dir = os.path.join(scan_dir, 'segmentation_masks')

    # Crop images
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('train')])
    img_fns = [img_fns[i] for i in indices]
    images = [Image.open(os.path.join(img_dir, img_fn)) for img_fn in img_fns]
    segs = [Image.open(os.path.join(seg_dir, img_fn)) for img_fn in img_fns]
    
    return concatenate_imgs(images, segs, ratio=ratio, pixel_sep=pixel_sep)
