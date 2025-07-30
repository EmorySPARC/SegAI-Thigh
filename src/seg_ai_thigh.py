#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet Thigh MRI Segmentation Wrapper

Author: Augustin C. Ogier

Wrapper script for nnU-Net v2 inference and post-processing.
Requires nnU-Net data folder layout under `nnunet_dir`:
  - nnUNet_raw
  - nnUNet_preprocessed
  - nnUNet_results

Usage:
    python seg_ai_thigh.py -i <input_nii> -o <mask_out> -n <path_to_nnunetv2_data>
"""

#Import library
#================================================================================================================================================#
import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
import nibabel as nib
from datetime import datetime
from sklearn.cluster import KMeans
from skimage.measure import label as cc_label
#================================================================================================================================================#

# Functions
#================================================================================================================================================#
#------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(
        description="Run nnU-Net v2 inference + post-processing"
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to input anatomical NIfTI (.nii.gz)"
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Path to output segmentation NIfTI (.nii.gz)"
    )
    p.add_argument(
        "--nnunet_dir", "-n", required=True,
        help="Path to nnUNetv2_data folder (contains nnUNet_raw, nnUNet_preprocessed, nnUNet_results)"
    )
    return p.parse_args()
#------------------------------------------------------------------------------------------------------------------------------------------------#
def super_copy(src, dst):
    """Copy file, ignoring certain filesystem errors."""
    try:
        shutil.copy(src, dst)
    except OSError as e:
        if e.errno != 95:
            raise
#------------------------------------------------------------------------------------------------------------------------------------------------#
def run_nnunet(input_nii, output_mask, nnunet_dir):
    """Run nnUNetv2_predict on a single file, then copy result to output_mask."""
    # Set nnU-Net environment variables
    os.environ["nnUNet_raw"] = os.path.join(nnunet_dir, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(nnunet_dir, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(nnunet_dir, "nnUNet_results")
    os.environ["nnUNet_inference"] = os.path.join(nnunet_dir, "nnUNet_inference")

    # Prepare temporary staging dirs
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    raw_dir = os.path.join(os.environ["nnUNet_inference"], ts, "Anat")
    out_dir = os.path.join(os.environ["nnUNet_inference"], ts, "Mask")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Stage input as <caseID>_0000.nii.gz
    case_id = os.path.basename(input_nii).replace(".nii.gz", "")
    staged = os.path.join(raw_dir, f"{case_id}_0000.nii.gz")
    super_copy(input_nii, staged)

    # Call nnUNetv2_predict
    cmd = [
        "nnUNetv2_predict",
        "-i", raw_dir,
        "-o", out_dir,
        "-d", "812",
        "-c", "3d_fullres",
        "-chk", "checkpoint_best.pth",
        "-f", "0"
    ]
    print(f"[nnUNet] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print(proc.stderr.decode(), file=sys.stderr)
        sys.exit(1)

    # Find the .nii.gz result
    preds = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
    if not preds:
        print("ERROR: No segmentation output found", file=sys.stderr)
        sys.exit(1)
    seg_path = os.path.join(out_dir, preds[0])

    # Copy to user-specified output
    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
    super_copy(seg_path, output_mask)
    print(f"[nnUNet] Segmentation saved -> {output_mask}")

    # Cleanup temp dirs
    try:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as e:
        print(f"[INFO] Could not remove temp folders: {e}", file=sys.stderr)
#------------------------------------------------------------------------------------------------------------------------------------------------#
def find_x_split(data):
    """Determine mid-sagittal plane to swap left/right labels."""
    coords = np.argwhere(data != 0)
    if coords.size == 0:
        return data.shape[0] // 2
    centroids = KMeans(2, random_state=0).fit(coords).cluster_centers_
    x_mid = centroids[:, 0].mean()
    all_x = np.arange(data.shape[0])
    mask_x = np.unique(coords[:, 0])
    zero_x = np.setdiff1d(all_x, mask_x)
    if zero_x.size:
        return int(zero_x[np.argmin(abs(zero_x - x_mid))])
    return int(round(x_mid))
#------------------------------------------------------------------------------------------------------------------------------------------------#
def keep_largest_component_per_label(data):
    """For each label, keep only the largest connected component."""
    out = np.zeros_like(data)
    for lab in np.unique(data[data != 0]):
        mask = (data == lab).astype(int)
        ccs = cc_label(mask)
        if ccs.max() > 1:
            counts = np.bincount(ccs.ravel())
            largest = counts[1:].argmax() + 1
            out[ccs == largest] = lab
        else:
            out[mask == 1] = lab
    return out
#------------------------------------------------------------------------------------------------------------------------------------------------#
def postprocess_mask(input_mask, output_mask, labels_right, labels_left):
    """Swap left/right where needed and remove small components."""
    vol = nib.load(input_mask)
    data = vol.get_fdata().astype(int)
    x_thr = find_x_split(data)
    print(f"[Postproc] Splitting at x={x_thr}")

    X = np.arange(data.shape[0])[:, None, None]
    # Swap right->left
    for r, l in zip(labels_right, labels_left):
        data[(data == r) & (X < x_thr)] = l
    # Swap left->right
    for l, r in zip(labels_left, labels_right):
        data[(data == l) & (X > x_thr)] = r

    data = keep_largest_component_per_label(data)
    nib.save(nib.Nifti1Image(data, vol.affine, vol.header), output_mask)
    print(f"[Postproc] Refined mask -> {output_mask}")
#------------------------------------------------------------------------------------------------------------------------------------------------#
#================================================================================================================================================#

# Main
#================================================================================================================================================#
def main():

    args = parse_args()

    # Step 1: nnU-Net inference
    temp_out = args.output.replace(".nii.gz", "_tmp.nii.gz")
    run_nnunet(args.input, temp_out, args.nnunet_dir)

    # Step 2: Post-processing
    labels_right = [5, 4, 3, 7, 6, 1, 2]
    labels_left  = [12, 11, 10, 14, 13, 8, 9]
    postprocess_mask(temp_out, args.output, labels_right, labels_left)

    # Cleanup
    if os.path.exists(temp_out):
        os.remove(temp_out)
    print("=== [Mask creation] All done. ===")

#------------------------------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
#================================================================================================================================================#