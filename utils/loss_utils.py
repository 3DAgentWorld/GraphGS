#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np
from PIL import Image
import os
from utils.graphics_utils import fov2focal

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_4x4(R, T):
    """
    :param R: 3x3
    :param T: 3,
    :return:  4x4
    """
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = R
    full_matrix[:3, 3] = T
    return full_matrix


def compute_transform(R1, T1, R2, T2):
    """
    Compute transformation matrix from view 1 to view 2
    Args:
        R1: Rotation matrix of view 1 (3x3)
        T1: Translation vector of view 1 (3,)
        R2: Rotation matrix of view 2 (3x3)
        T2: Translation vector of view 2 (3,)
    Returns:
        transformation_matrix: 4x4 transformation matrix from view 1 to view 2
    """
    # Convert camera poses to world-to-camera transformation matrices
    w2c1 = get_4x4(R1.transpose(), T1)
    w2c2 = get_4x4(R2.transpose(), T2)
    
    # Compute the transformation matrix from view 1 to view 2
    # First convert view 1 to world coordinates (inverse of w2c1)
    # Then convert from world to view 2 coordinates (w2c2)
    transformation_matrix = w2c2 @ np.linalg.inv(w2c1)
    transformation_matrix = np.linalg.inv(transformation_matrix)
    return transformation_matrix

def get_4x4_c2w(R, T):
    # Directly construct homogeneous matrix when R and T are already in camera-to-world format
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = R
    full_matrix[:3, 3] = T
    return full_matrix

def compute_transform_c2w(R1, T1, R2, T2):
    """
    Compute transformation matrix from view 1 to view 2 when input (R, T) are already c2w matrices
    For a point p1 in view 1 (in camera 1 coordinate system), we first transform it to world coordinates:
        p_world = c2w1 * p1,
    then transform from world coordinates to camera 2 coordinates:
        p2 = (c2w2)^{-1} * p_world.
    Therefore:
        transformation_matrix = (c2w2)^{-1} * c2w1
    """
    c2w1 = get_4x4_c2w(R1, T1)
    c2w2 = get_4x4_c2w(R2, T2)
    transformation_matrix = np.linalg.inv(c2w2) @ c2w1
    return transformation_matrix


def compute_homography(K1, K2, T_2_1, d=1.0):
    """
    Compute the homography matrix based on camera intrinsics and 3D rigid transformation between views.
    Assumes all pixels lie on a plane at z=d.
    Args:
        K1: Intrinsic matrix of image 1 (camera 1) (3x3)
        K2: Intrinsic matrix of image 2 (camera 2) (3x3)
        T_2_1: 4x4 rigid transformation matrix from camera 1 to camera 2
        d: Plane depth (assumed to be at z=d)
    Returns:
        H: 3x3 homography matrix
    """
    R = T_2_1[:3, :3]
    t = T_2_1[:3, 3].reshape(3, 1)
    n = np.array([0, 0, 1]).reshape(3, 1) 
    H = K2 @ (R - t @ n.T / d) @ np.linalg.inv(K1)
    return H


def debug_save_and_show(current_image, warped_image, name="debug"):
    """
    Save and display alignment between current and warped images
    Args:
        current_image: current view image tensor (C,H,W)
        warped_image: warped previous view image tensor (C,H,W)
        name: prefix for saved files
    """
    debug_dir = "debug_alignment"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    current_np = current_image.detach().cpu().numpy()
    warped_np = warped_image.detach().cpu().numpy()
    
    print(f"Debug - current_np shape: {current_np.shape}, dtype: {current_np.dtype}")
    print(f"Debug - warped_np shape: {warped_np.shape}, dtype: {warped_np.dtype}")
    
    # Transpose from (C,H,W) to (H,W,C)
    current_np = np.transpose(current_np, (1,2,0))
    warped_np = np.transpose(warped_np, (1,2,0))
    
    print(f"Debug - After transpose - current_np shape: {current_np.shape}")
    
    # Ensure both images have the same shape
    if current_np.shape != warped_np.shape:
        print(f"Warning: Image shapes do not match: {current_np.shape} vs {warped_np.shape}")
        # Resize warped_np to match current_np if needed
        if current_np.shape[0] > 0 and current_np.shape[1] > 0:
            from skimage.transform import resize
            warped_np = resize(warped_np, (current_np.shape[0], current_np.shape[1], current_np.shape[2]),
                               anti_aliasing=True, preserve_range=True)
    
    # Scale to 0-255 range if needed
    if current_np.max() <= 1.0:
        current_np = (current_np * 255)
    current_np = current_np.clip(0, 255).astype(np.uint8)
    
    if warped_np.max() <= 1.0:
        warped_np = (warped_np * 255)
    warped_np = warped_np.clip(0, 255).astype(np.uint8)
    
    # Verify shapes again
    print(f"Debug - Final shapes: current_np {current_np.shape}, warped_np {warped_np.shape}")
    
    try:
        # Create overlay directly
        overlay_np = (current_np * 0.5 + warped_np * 0.5).astype(np.uint8)
        
        # Save individual images as numpy arrays
        np.save(os.path.join(debug_dir, f"{name}_current.npy"), current_np)
        np.save(os.path.join(debug_dir, f"{name}_warped.npy"), warped_np)
        
        # Convert numpy arrays to PIL images
        current_img = Image.fromarray(current_np)
        warped_img = Image.fromarray(warped_np)
        overlay_img = Image.fromarray(overlay_np)
        
        # Create comparison image
        width = current_np.shape[1]
        height = current_np.shape[0]
        comparison = Image.new('RGB', (width * 3, height))
        comparison.paste(current_img, (0, 0))
        comparison.paste(warped_img, (width, 0))
        comparison.paste(overlay_img, (width * 2, 0))
        
        # Save images
        comparison.save(os.path.join(debug_dir, f"{name}_comparison.png"))
        current_img.save(os.path.join(debug_dir, f"{name}_current.png"))
        warped_img.save(os.path.join(debug_dir, f"{name}_warped.png"))
        overlay_img.save(os.path.join(debug_dir, f"{name}_overlay.png"))
        
        print(f"Debug images saved to {debug_dir} with prefix '{name}'")
    except Exception as e:
        print(f"Error in debug_save_and_show: {e}")
        print(f"current_np shape: {current_np.shape}, dtype: {current_np.dtype}, min: {current_np.min()}, max: {current_np.max()}")
        print(f"warped_np shape: {warped_np.shape}, dtype: {warped_np.dtype}, min: {warped_np.min()}, max: {warped_np.max()}")
        # Save raw data for later inspection
        np.save(os.path.join(debug_dir, f"{name}_current_error.npy"), current_np)
        np.save(os.path.join(debug_dir, f"{name}_warped_error.npy"), warped_np)

# def warp_image(img, transform_matrix):

#     img_array = np.array(img)

#     height = img_array.shape[1]
#     width = img_array.shape[2]
#     img_array = np.transpose(img_array, (1, 2, 0))
#     transform_matrix = transform_matrix.astype(np.float32)[:3, :3]

#     warped_img = cv2.warpPerspective(img_array, transform_matrix, (width, height))
#     warped_img = (warped_img * 255).astype(np.uint8)
#     return warped_img

def warp_image(img, H):
    img_array = np.array(img)

    height = img_array.shape[1]
    width = img_array.shape[2]
    img_array = np.transpose(img_array, (1, 2, 0))
    warped_img = cv2.warpPerspective(img_array, H.astype(np.float32), (width, height))
    warped_img = (warped_img * 255).astype(np.uint8)
    return warped_img


def overlay_images(img1, img2, alpha=0.5):
    """
    Overlay two images with alpha blending
    Args:
        img1: first image array (H,W,C)
        img2: second image array (H,W,C)
        alpha: blending factor
    Returns:
        overlay image
    """
    # Ensure both images have the same shape
    assert img1.shape == img2.shape, f"Image shapes do not match: {img1.shape} vs {img2.shape}"
    
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    # Don't transpose if already in (H,W,C) format
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        pass
    elif len(img1.shape) == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        pass
    elif len(img2.shape) == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Scale to 0-255 range if needed
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)
        
    overlay = (img1 * alpha + img2 * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay)

def show_image(image, window_name='Image'):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def compute_multiview_consistency_loss(current_image, last_image, last_view, current_view):
    """
    Compute multi-view consistency loss
    Args:
        current_image: rendered image from current viewpoint
        last_image: rendered image from previous viewpoint
        last_view: camera parameters of previous viewpoint
        current_view: camera parameters of current viewpoint
    Returns:
        consistency_loss: consistency loss between two views
        mask: valid region mask where both images have content
    """
    #print(f"Current image shape: {current_image.shape}, Last image shape: {last_image.shape}")
    
    # Check if images are too small for meaningful debug visualization
    if current_image.shape[1] <= 1 or current_image.shape[2] <= 1:
        print("Warning: Image dimensions too small for visualization, skipping debug")
    else:
        try:
            current_width, current_height = current_view.image_width, current_view.image_height
            current_focal_x = fov2focal(current_view.FoVx, current_width)
            current_focal_y = fov2focal(current_view.FoVy, current_height)
            K = np.array([
                [current_focal_x, 0, current_width/2],
                [0, current_focal_y, current_height/2],
                [0, 0, 1]
            ])

            last_width, last_height = last_view.image_width, last_view.image_height
            last_focal_x = fov2focal(last_view.FoVx, last_width)
            last_focal_y = fov2focal(last_view.FoVy, last_height)
            K_last = np.array([
                [last_focal_x, 0, last_width/2],
                [0, last_focal_y, last_height/2],
                [0, 0, 1]
            ])      

            transform_matrix = compute_transform_c2w(last_view.R, last_view.T, current_view.R, current_view.T)
            homography = compute_homography(K_last, K, transform_matrix)
            warped_img = np.transpose(warp_image(last_image, homography), (2, 0, 1)) / 255
            warped_img = torch.tensor(warped_img).cuda()
            
            # Calculate mask for valid regions
            mask1 = (warped_img > 0)
            mask2 = (current_image > 0)
            mask = mask1 * mask2
            
            # Compute consistency loss
            consistency_loss = l1_loss(current_image, warped_img) * mask
            
            return consistency_loss, mask
        except Exception as e:
            print(f"Error in compute_multiview_consistency_loss: {e}")
            # Return zero loss and full mask as fallback
            mask = torch.ones_like(current_image).bool()
            return torch.tensor(0.0, device=current_image.device), mask
    
    # If images are too small, return zero loss
    mask = torch.ones_like(current_image).bool()
    return torch.tensor(0.0, device=current_image.device), mask