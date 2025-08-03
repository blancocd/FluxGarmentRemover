import os
import sys
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from tqdm import tqdm
from utils.create_masks_from_seg import get_mask_4ddress


fourddress_palette = np.array([
    [128., 128., 128.],   # 0 skin
    [255., 128.,   0.],   # 1 hair
    [128.,   0., 255.],   # 2 shoes
    [255.,   0.,   0.],   # 3 inner
    [  0., 255.,   0.],   # 4 lower
    [  0., 128., 255.],   # 5 outer
]).astype(np.uint8)

segformer_to_4ddress = [
    [1, 3, 11, 12, 13, 14, 15], # 0: skin
    [2],                        # 1: hair
    [9, 10],                    # 2: shoes
    [4, 7, 16, 17],             # 3: inner
    [5, 6, 8],                  # 4: lower
    []                          # 5: outer
]

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    img_fns = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

    # Load SegFormer model
    processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda")

    # Segment generated images
    for img_fn in tqdm(img_fns):
        image = Image.open(os.path.join(directory, img_fn))
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            upsampled = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            pred = upsampled.argmax(dim=1)[0].cpu().numpy()

        H, W = pred.shape
        segmentation_map = np.zeros((H, W, 3), dtype=np.uint8)
        for label, seg_former_labels in enumerate(segformer_to_4ddress):
            for seg_former_label in seg_former_labels:
                segmentation_map[pred==seg_former_label] = fourddress_palette[label]

        # Save segmentation map, inner and lower masks
        os.makedirs(os.path.join(directory, "segformer_segmentation_masks"), exist_ok=True)
        segmentation_map_path = os.path.join(directory, "segformer_segmentation_masks", img_fn)
        Image.fromarray(segmentation_map).save(segmentation_map_path)

        inner_mask = get_mask_4ddress(segmentation_map, 'inner', dil_its=0, ero_its=None)
        inner_mask_pil = Image.fromarray((inner_mask * 255).astype(np.uint8))
        os.makedirs(os.path.join(directory, "inner_mask"), exist_ok=True)
        inner_mask_path = os.path.join(directory, "inner_mask", img_fn)
        inner_mask_pil.save(inner_mask_path)

        lower_mask = get_mask_4ddress(segmentation_map, 'lower', dil_its=0, ero_its=None)
        lower_mask_pil = Image.fromarray((lower_mask * 255).astype(np.uint8))
        os.makedirs(os.path.join(directory, "lower_mask"), exist_ok=True)
        lower_mask_path = os.path.join(directory, "lower_mask", img_fn)
        lower_mask_pil.save(lower_mask_path)

if __name__ == "__main__":
    main()
