import argparse
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity
import cv2
import numpy as np

def load_mask(filename):
    img = cv2.imread(filename)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(bool)
    return mask

def main(results_dirs, results_fns, garment_data_json):
    print(f"Evaluating {results_dirs} and saving results to {results_fns}.")
    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    gen_dir = '/mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/GEN'

    view = 'flbr'
    images_dir = '/mnt/lustre/work/ponsmoll/pba534/ffgarments/data/images/' + view
    upper_masks_dir = '/mnt/lustre/work/ponsmoll/pba534/ffgarments/data/masks/dil7_er2/upper'
    outer_masks_dir = '/mnt/lustre/work/ponsmoll/pba534/ffgarments/data/masks/dil7_er2/outer'

    with open('/mnt/lustre/work/ponsmoll/pba534/ffgarments/data/captions/scan_testset.json', 'r') as f:
        scans = json.load(f)

    results_lists = []
    for results_fn in results_fns:
        if os.path.isfile(results_fn):
            with open(results_fn, 'r') as f:
                results_lists.append(json.load(f))
        else:
            results_lists.append(list(range(101)))

    specific_scans = range(100) #[10,11,13,15,16,21,24,25,27,40,45,47,55,56,61,68,73,75,79,81,85,86,89,92,96,97,98,99]
    scans = [scans[i] for i in specific_scans]
    for results_dir, results_fn, results_list in zip(results_dirs, results_fns, results_lists):
        for scan_id, scan in tqdm(zip(specific_scans, scans), total=len(specific_scans)):
            inner_garm = scan['inner']
            outer_garm = scan['outer']
            outer_garm_type = outer_garm.split(' ')[-1].replace('-', ' ')
            outer_garm_types = ['jacket', 'cardigan', 'sweater', 'coat']
            outer_garm_types = [outer_garm_type] + [ogt for ogt in outer_garm_types if ogt not in outer_garm_type]

            orig_img_fn = os.path.join(images_dir, f'{scan_id}.png')

            if len(outer_garm_types) > 1:
                outer_garm_types_str = ', '.join(outer_garm_types[:-1]) + f', or {outer_garm_types[-1]}'
            else:
                outer_garm_types_str = outer_garm_types[0]

            prompt_short_outer_desc = f'Answer yes if the person in the image is wearing a {outer_garm_type}. Answer no if the person is only wearing a {inner_garm}, shirt, t-shirt, or blouse.'
            prompt_long_outer_desc = f'Answer yes if the person in the image is wearing a {outer_garm_types_str}. Answer no if the person is only wearing a {inner_garm}, shirt, t-shirt, or blouse.'
        
            answers = []
            flux_img_fn = os.path.join(results_dir, f'{scan_id}.png')
            for prompt in [prompt_short_outer_desc, prompt_long_outer_desc]:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description for top body garments."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": flux_img_fn},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
                    generation = generation[0][input_len:]

                decoded = processor.decode(generation, skip_special_tokens=True)
                answers.append(decoded)

            # The short prompt is more reliable. When the answers to the short and long prompts disagree, it is worth seeing 
            # why is this, still the long prompt is too strict and is wrong around 7/10 of the time while the short prompt 3/10
            # So we use this same weights to acount for disagreements.
            vlm_score = 0.7*int('no' in answers[0].lower()) + 0.3*int('no' in answers[1].lower())

            orig_img = cv2.imread(orig_img_fn)
            flux_img = cv2.imread(flux_img_fn)
            upper_mask = load_mask(os.path.join(upper_masks_dir, f'{scan_id}.png'))
            outer_mask = load_mask(os.path.join(outer_masks_dir, f'{scan_id}.png'))
            shouldnt_have_edited_mask = np.logical_and(upper_mask, ~outer_mask)

            _, ssim_map = structural_similarity(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY), 
                                                cv2.cvtColor(flux_img, cv2.COLOR_BGR2GRAY), full=True)
            ssim_masked = np.mean(ssim_map[shouldnt_have_edited_mask])

            squared_diff_map = (orig_img.astype(np.float32)/255. - flux_img.astype(np.float32)/255.) ** 2
            mse_masked = np.mean(squared_diff_map[shouldnt_have_edited_mask])

            psnr_masked = 10 * np.log10(1. / mse_masked)

            result_dict = {
                'short' : {
                    'prompt': prompt_short_outer_desc,
                    'full_answer' : answers[0],
                    'successful': 'no' in answers[0].lower()
                },
                'long' : {
                    'prompt': prompt_long_outer_desc,
                    'full_answer' : answers[1],
                    'successful': 'no' in answers[1].lower()
                },
                'vlm_score': f"{vlm_score:.4f}",
                'ssim_masked': f"{ssim_masked:.4f}",
                'mse_masked': f"{mse_masked:.6f}",
                'psnr_masked': f"{psnr_masked:.2f}"
                }
            results_list[scan_id] = result_dict

        vlm_scores = [float(results_dict['vlm_score']) for results_dict in results_list[:-1]]
        vlm_scores_short = [int(results_dict['short']['successful']) for results_dict in results_list[:-1]]
        vlm_scores_long = [int(results_dict['long']['successful']) for results_dict in results_list[:-1]]
        ssims = [float(results_dict['ssim_masked']) for results_dict in results_list[:-1]]
        mses = [float(results_dict['mse_masked']) for results_dict in results_list[:-1]]
        psnrs = [float(results_dict['psnr_masked']) for results_dict in results_list[:-1]]

        print("==== Evaluation Report ====")
        print(f"Mean VLM Score: {np.mean(vlm_scores):.4f}")
        print(f"Sum VLM Score: {np.sum(vlm_scores)}")
        print(f"Mean SSIM (masked): {np.mean(ssims):.4f}")
        print(f"Mean MSE (masked): {np.mean(mses):.6f}")
        print(f"Mean PSNR (masked): {np.mean(psnrs):.2f}")
        print("==========================")

        # Add summary as the last entry in results_list
        results_list[-1] = {
            'mean_vlm_score': f"{np.mean(vlm_scores):.4f}",
            'mean_vlm_score_short_prompt': f"{np.mean(vlm_scores_short):.4f}",
            'mean_vlm_score_long_prompt': f"{np.mean(vlm_scores_long):.4f}",
            'mean_ssim_masked': f"{np.mean(ssims):.4f}",
            'mean_mse_masked': f"{np.mean(mses):.6f}",
            'mean_psnr_masked': f"{np.mean(psnrs):.2f}"
        }

        with open(results_fn, 'w') as f:
            json.dump(results_list, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate inpainting results.")
    parser.add_argument("--results_dir", nargs='+', required=True, help="Directories with result images")
    parser.add_argument("--results_fn", nargs='+', required=True, help="Filenames for output JSON results")
    parser.add_argument("--garment_data_json", required=True, help="JSON with prompts per scan")
    args = parser.parse_args()
    main(args.results_dir, args.results_fn, args.garment_data_json)
