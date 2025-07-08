import json
import numpy as np
import os

import matplotlib.pyplot as plt

json_files = [
    "/mnt/lustre/work/ponsmoll/pba534/ffgarments/vae_tests/vae_flux_fill_results.json",
    "/mnt/lustre/work/ponsmoll/pba534/ffgarments/vae_tests/vae_flux_kontext_results.json",
    "/mnt/lustre/work/ponsmoll/pba534/ffgarments/vae_tests/vae_sdxl_results.json"
]

def load_json_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def compute_stats(values):
    arr = np.array(values, dtype=np.float32)
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'max': np.max(arr),
        'min': np.min(arr)
    }

def print_report(name, psnr_stats, ssim_stats):
    print(f"Report for {name}:")
    print(f"  PSNR: mean={psnr_stats['mean']:.4f}, median={psnr_stats['median']:.4f}, max={psnr_stats['max']:.4f}, min={psnr_stats['min']:.4f}")
    print(f"  SSIM: mean={ssim_stats['mean']:.4f}, median={ssim_stats['median']:.4f}, max={ssim_stats['max']:.4f}, min={ssim_stats['min']:.4f}")
    print()

for json_file in json_files:
    data = load_json_data(json_file)
    # Sort by idx to ensure correct order
    data_sorted = sorted(data, key=lambda d: d['idx'])
    indices = [entry['idx'] for entry in data_sorted]
    psnrs = [float(entry['psnr']) for entry in data_sorted]
    ssims = [float(entry['ssim']) for entry in data_sorted]

    psnr_stats = compute_stats(psnrs)
    ssim_stats = compute_stats(ssims)
    print_report(os.path.basename(json_file), psnr_stats, ssim_stats)
