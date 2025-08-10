import os
import gc
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

def segment_dir(directory):
    img_dir = os.path.join(directory, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

    # Load SegFormer model
    processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda")

    # Segment generated images
    for img_fn in tqdm(img_fns):
        image = Image.open(os.path.join(img_dir, img_fn))
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            upsampled = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        del inputs; del logits; del upsampled

        H, W = pred.shape
        segmentation_map = np.zeros((H, W, 3), dtype=np.uint8)
        for label, seg_former_labels in enumerate(segformer_to_4ddress):
            for seg_former_label in seg_former_labels:
                segmentation_map[pred==seg_former_label] = fourddress_palette[label]
        del pred

        # Save segmentation map
        os.makedirs(os.path.join(directory, "segmentation_masks"), exist_ok=True)
        segmentation_map_path = os.path.join(directory, "segmentation_masks", img_fn)
        Image.fromarray(segmentation_map).save(segmentation_map_path)
        gc.collect(); torch.cuda.empty_cache()
    
    del processor; del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <directory>")
        sys.exit(1)
    segment_dir(sys.argv[1])
