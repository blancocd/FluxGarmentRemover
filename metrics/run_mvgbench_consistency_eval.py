import os
import json
import shutil
import argparse
import subprocess

def prepare_data(scan_dir):
    print(f"Preparing data for scan: {scan_dir}")

    odd_dir = os.path.join(scan_dir, "odd_views")
    even_dir = os.path.join(scan_dir, "even_views")
    os.makedirs(os.path.join(odd_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(even_dir, "images"), exist_ok=True)

    transforms_path = os.path.join(scan_dir, "transforms_train.json")
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Separate frames into odd and even lists
    odd_frames = []
    even_frames = []
    for i, frame in enumerate(transforms["frames"]):
        # Extract the image number from the file path
        img_num = int(os.path.basename(frame["image_path"]).split("_")[-1])

        # Get the original image path
        original_img_path = os.path.join(scan_dir, "images", f"gen_{img_num:04d}.png")
        
        # Determine the new image path and frame list
        if i % 2 == 0:
            new_img_path = os.path.join(even_dir, "images", f"train_{i//2:04d}.png")
            frame["image_path"] = f"./images/train_{i//2:04d}.png"
            even_frames.append(frame)
        else:
            new_img_path = os.path.join(odd_dir, "images", f"train_{i//2:04d}.png")
            frame["image_path"] = f"./images/train_{i//2:04d}.png"
            odd_frames.append(frame)
            
        # Copy and rename the image
        shutil.copy(original_img_path, new_img_path)

    # Create new transforms for odd and even views
    odd_transforms = {**transforms, "frames": odd_frames}
    even_transforms = {**transforms, "frames": even_frames}

    # Write the new transforms to JSON files
    with open(os.path.join(odd_dir, "transforms_train.json"), "w") as f:
        json.dump(odd_transforms, f, indent=4)
    with open(os.path.join(even_dir, "transforms_train.json"), "w") as f:
        json.dump(even_transforms, f, indent=4)

    print("Data preparation complete.")

def run_mvfit(mvg_bench_dir, scan_dir):
    print("Running 3DGS fitting for both odd and even views")

    # Run mvfit for even views
    even_views_path = os.path.join(scan_dir, "even_views")
    subprocess.run([
        "python", os.path.join(mvg_bench_dir, "run_mvfit.py"),
        even_views_path, "--white_background"
    ], check=True, cwd=mvg_bench_dir)

    # Run mvfit for odd views
    odd_views_path = os.path.join(scan_dir, "odd_views")
    subprocess.run([
        "python", os.path.join(mvg_bench_dir, "run_mvfit.py"),
        odd_views_path, "--white_background"
    ], check=True, cwd=mvg_bench_dir)

    print("3DGS fitting complete.")

def run_evaluation(mvg_bench_dir, scan_dir):
    print("Running 3D consistency evaluation...")

    scan_name = os.path.basename(scan_dir)
    odd_output_name = f"output/consistency/{scan_name}_odd_views"
    even_output_name = f"output/consistency/{scan_name}_even_views"

    subprocess.run([
        "python", os.path.join(mvg_bench_dir, "eval", "eval_consistency.py"),
        "--name_odd", odd_output_name,
        "--name_even", even_output_name
    ], check=True, cwd=mvg_bench_dir)

    print("3D consistency evaluation complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Run MVGBench's 3D consistency evaluation pipeline."
    )
    parser.add_argument(
        "--scans_dir", type=str, required=True,
        help="The parent directory containing all the scan subdirectories."
    )
    parser.add_argument(
        "--scan_index", type=int, required=True,
        help="The index of the scan to process (for array jobs)."
    )
    parser.add_argument(
        "--mvg_bench_dir", type=str, required=True,
        help="The path to the MVGBench repository."
    )
    parser.add_argument(
        "--skip_data_prep", action="store_true",
        help="Skip the data preparation step."
    )
    parser.add_argument(
        "--skip_mvfit", action="store_true",
        help="Skip the 3DGS fitting step."
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip the 3D consistency evaluation step."
    )
    args = parser.parse_args()

    # Get the selected scan directory
    scans = sorted([os.path.join(args.scans_dir, d) for d in os.listdir(args.scans_dir) if os.path.isdir(os.path.join(args.scans_dir, d))])
    scan_dir = scans[args.scan_index]

    # Run the pipeline
    if not args.skip_data_prep:
        prepare_data(scan_dir)
    if not args.skip_mvfit:
        run_mvfit(args.mvg_bench_dir, scan_dir)
    if not args.skip_eval:
        run_evaluation(args.mvg_bench_dir, scan_dir)

if __name__ == "__main__":
    main()
