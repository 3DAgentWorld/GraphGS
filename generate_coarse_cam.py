from VisAnything.coordEngine.read_colmap import read_colmap
from VisAnything.coordEngine.read_waymo import read_waymo
from VisAnything.coordEngine.read_nerf_syn import read_nerf_syn
import sys
import os
import json
import torch
import argparse


def detect_format_and_read(path):
    """
    Detect the format of camera data and read accordingly
    Supported formats:
    - COLMAP (contains 'sparse' directory)
    - NeRF Synthetic (contains 'transforms_train.json')
    - Waymo (contains 'scenario.pt')
    - Dust3r (fallback option)
    
    Args:
        path (str): Path to the dataset directory
        
    Returns:
        Camera data object or None if reading fails
    """
    if os.path.exists(os.path.join(path, "sparse")):
        print("Detected COLMAP format")
        return read_colmap(colmap_path=path)
    elif os.path.exists(os.path.join(path, "transforms_train.json")):
        print("Detected NeRF Synthetic format")
        return read_nerf_syn(path)
    elif os.path.exists(os.path.join(path, "scenario.pt")):
        print("Detected Waymo format")
        return read_waymo(path)
    else:
        print("No known format detected, initializing as Dust3r format")
        return None  # TODO: implement Dust3r format reading


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate coarse camera poses from various formats')
    parser.add_argument('--path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()

    # Read camera data based on detected format
    all_cams = detect_format_and_read(args.path)
    
    if all_cams is not None:
        print(f'Number of cameras read: {len(all_cams)}')
        for cam in all_cams:
            print(cam.image_path)
        all_cams.save_to_json(target_path=args.path)
    else:
        print("Failed to read camera data")