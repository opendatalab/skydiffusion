

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import ImageOps

def grid_sample(image, optical, jac=None):
    # Interpolation function 
    C, IH, IW = image.shape # Extracting dimensions from the image tensor
    image = image.unsqueeze(0)

    _, H, W, _ = optical.shape  # Extracting dimensions from the optical tensor

    ix = optical[..., 0].view(1, 1, H, W)  
    iy = optical[..., 1].view(1, 1, H, W)  

    with torch.no_grad():
        ix_nw = torch.floor(ix)  
        iy_nw = torch.floor(iy)  
        ix_ne = ix_nw + 1        
        iy_ne = iy_nw            
        ix_sw = ix_nw            
        iy_sw = iy_nw + 1        
        ix_se = ix_nw + 1        
        iy_se = iy_nw + 1       

        # Clamp coordinates to be within valid range
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    # Create masks for valid coordinates
    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0  # Ensure that there are valid coordinates

    # Calculate the weights for interpolation
    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    # Flatten the image for easier indexing
    image = image.view(1, C, IH * IW)

    # Gather the values at the four corners
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)

    # Perform bilinear interpolation
    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:
        # Calculate the gradients with respect to x and y
        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # Combine with the jacobian if provided
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        return out_val, jac_new1  # Return the interpolated values and updated jacobian
    else:
        return out_val, None  # Return only the interpolated values if no jacobian is provided


def BEV_transform(rot, S, H, W, meter_per_pixel, Camera_height):

    # This function performs BEV conversion and establishes the mapping relationships between different coordinates.
    # Create a meshgrid for coordinates
    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=rot.device), 
                            torch.arange(0, S, dtype=torch.float32, device=rot.device), indexing='ij')
    ii = ii.unsqueeze(dim=0).repeat(1, 1, 1)  # Expand and repeat for batch size
    jj = jj.unsqueeze(dim=0).repeat(1, 1, 1)  # Expand and repeat for batch size
    
    # h = 0  # Flat Earth Hypothesis (fixed at 0)
    # Cruved-BEV transformationß
    max_dist = torch.sqrt(torch.tensor(2*(S/2)**2, dtype=torch.float32, device=rot.device))
    dist_to_center = torch.sqrt((ii - S/2)**2 + (jj - S/2)**2)
    normalized_dist = dist_to_center / max_dist
    h = 3 * (normalized_dist**4)


    # Calculate the radius from the center of the grid
    radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)
    
    # Calculate the angle (theta) in radians
    theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    
    # Normalize theta to be within the range [0, 2π)
    theta = (-np.pi / 2 + theta % (2 * np.pi)) % (2 * np.pi)
    
    # Adjust theta by the rotation and map to image width (W) 
    theta = (theta + rot[:, None, None] * np.pi) % (2 * np.pi)
    theta = theta / (2 * np.pi) * W
    
    # Calculate the minimum elevation angle (phimin)
    meter_per_pixel_tensor = torch.full((1, 1, 1), meter_per_pixel, device=radius.device)
    phimin = torch.atan2(radius * meter_per_pixel_tensor, torch.tensor(Camera_height, device=radius.device) + h)
    phimin = phimin / np.pi * H  # Map to image height (H)

    # Stack theta and phimin to create UV coordinates
    uv = torch.stack([theta, phimin], dim=-1)
    
    return uv  # Return the UV coordinates

def resize_and_pad_image(image, target_height, target_width):

    # For CVUSA
    # Get the current dimensions of the image
    current_width, current_height = image.size

    # Calculate the padding needed for height
    if current_height == target_height:
        padded_image = image
    else:
        # Calculate the padding needed for the top and bottom
        top_padding = (target_height - current_height) // 2
        bottom_padding = target_height - current_height - top_padding
        # Use Pillow's ImageOps.expand to add padding
        padded_image = ImageOps.expand(image, (0, top_padding, 0, bottom_padding), fill='black')

    # Resize the image to the target dimensions
    resized_image = padded_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_image

def Transformation_SVI(img_path, uv, H, W, resize_and_pad=False):
    # This function performs BEV conversion in batch mode.

    # Define transformation: resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])
    

    # Process images 

    # Load and transform images
    image = Image.open(img_path)

    # Apply resize and pad preprocessing if the flag is set to True
    if resize_and_pad:
        image = resize_and_pad_image(image, H, W)

    image_tensor = transform(image)

    # Apply grid sampling transformation
    transformed_image, _ = grid_sample(image_tensor, uv)

    return np.transpose(transformed_image.detach().cpu().numpy()[0], (1, 2, 0))


def CruvedBEV(input_path, dataset="CVACT"):

    # Set parameters for image processing（CVACT）
    if dataset == "CVACT":
        S = 512           # Size parameter for the grid (Satellite size)
        H = 832           # Height of the input street image
        W = 1664          # Width of the input street image
        Camera_height = -1.5 # Camera height parameter for BEV transformation (Assume the difference between the ground height and the camera height)
    
    elif dataset == "CVUSA":
        S = 512           # Size parameter for the grid (Satellite size)
        H = 616           # Height of the input street image
        W = 1232          # Width of the input street image
        Camera_height = -1.5 # Camera height parameter for BEV transformation (Assume the difference between the ground height and the camera height)
    
    else:
        raise ValueError("The dataset must be one of [CVACT, CVUSA, VIGOR, G2A-3]")

    # Create a rotation tensor with all values set to 90 degrees (If North is in the center of Street View)
    rot = torch.tensor([90], dtype=torch.float32)

    # Define the scale of meters per pixel （CVACT & CVUSA）
    meter_per_pixel = 0.06 

    # Compute UV coordinates for the satellite to ground transformation
    uv = BEV_transform(rot, S, H, W, meter_per_pixel, Camera_height)

    # Process images using the computed UV coordinates
    Transformation_SVI(input_path, uv, H, W)
