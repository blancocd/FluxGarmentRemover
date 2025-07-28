import os
import sys
import torch
import numpy as np
from PIL import Image
from glob import glob
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

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
        print(f"Usage: python {os.path.basename(__file__)} <scanname>")
        sys.exit(1)
    scan_name = sys.argv[1]

    print("Segmenting generated images")

    # Load SegFormer model
    processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda")

    # Segment generated images
    gen_paths = sorted(glob(os.path.join(scan_name, "images", "gen_*.png")))
    for gen_path in gen_paths:
        gen_image = Image.open(gen_path).convert("RGB")
        inputs = processor(images=gen_image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            upsampled = F.interpolate(logits, size=gen_image.size[::-1], mode="bilinear", align_corners=False)
            pred = upsampled.argmax(dim=1)[0].cpu().numpy()

        H, W = pred.shape
        segmentation_mask = np.zeros((H, W, 4), dtype=np.uint8)
        for label, seg_former_labels in enumerate(segformer_to_4ddress):
            for seg_former_label in seg_former_labels:
                segmentation_mask[pred==seg_former_label, :3] = fourddress_palette[label]
                segmentation_mask[pred==seg_former_label, 3] = 255

        # Save segmentation map
        os.makedirs(os.path.join(scan_name, "segmentation_masks"), exist_ok=True)
        segmentationmask_path = os.path.join(scan_name, "segmentation_masks", os.path.basename(gen_path))
        Image.fromarray(segmentation_mask).save(segmentationmask_path)

        # Set background to transparent, shouldn't use segmap as it's not accurate enough
        gen_np = np.array(gen_image)
        # Generated images have white background, if the pixel with most value is  more than 250
        # then it must be the background. 250 to leave some room
        bg_mask = np.min(gen_np, axis=-1) > 250
        alpha = (np.ones(gen_np.shape[:2]) * 255).astype(np.uint8)
        alpha[bg_mask] = 0
        gen_rgba = np.dstack((gen_np, alpha))
        Image.fromarray(gen_rgba).save(gen_path)

if __name__ == "__main__":
    main()
