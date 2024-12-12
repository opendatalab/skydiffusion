import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils
import pytorch_ssim
from skimage import io, color
from skimage.filters import roberts, sobel_h, sobel_v
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import kornia as K
from os import listdir
from os.path import isfile, join
import re
import cv2
from skimage import io
import lpips

import pytorch_fid

# Import GPU-related libraries
import torch.cuda

def PSNR(true_frame, pred_frame):
    eps = 0.0001
    prediction_error = 0.0
    [h, w, c] = true_frame.shape
    dev_frame = pred_frame - true_frame
    dev_frame = np.multiply(dev_frame, dev_frame)
    prediction_error = np.mean(dev_frame)
    if prediction_error > eps:
        prediction_error = 10 * np.log((1 * 1) / prediction_error) / np.log(10)
    else:
        prediction_error = 10 * np.log((1 * 1) / eps) / np.log(10)
    return prediction_error

def SSIM(img1, img2):
    [h, w, c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:, :, 0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:, :, 0]
    score = ssim(img1, img2, data_range=1.0)
    return score

def SSIM_RGB(img1, img2):
    score = ssim(img1, img2, data_range=1.0, multichannel=True)
    return score

def L1difference(img_true, img_pred):
    [h, w] = img_true.shape
    true_gx = sobel_h(img_true) / 4.0
    true_gy = sobel_v(img_true) / 4.0
    pred_gx = sobel_h(img_pred) / 4.0
    pred_gy = sobel_v(img_pred) / 4.0
    dx = np.abs(true_gx - pred_gx)
    dy = np.abs(true_gy - pred_gy)
    prediction_error = np.mean(dx + dy)
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10 * np.log((1 * 1) / prediction_error) / np.log(10)
    else:
        prediction_error = 10 * np.log((1 * 1) / eps) / np.log(10)
    return prediction_error

if __name__ == "__main":
    # Ensure that the GPU is available
    device = torch.device("cuda")

    # Your input arguments
    fold_pred = sys.argv[1].strip()
    fold_gt = sys.argv[2].strip()

    # Other variables
    PSNR_score_avg = 0
    SSIM_score_avg = 0
    L1_score_avg = 0
    dist_avg = 0
    file_list = [f for f in os.listdir(fold_pred) ]

    # Define the LPIPS model and move it to the GPU
    # loss_fn = lpips.LPIPS(net='alex')
    # loss_fn = loss_fn.to(device)

    width, height = 1024, 512
    i = 0

    for f in file_list:
        i += 1
        img_pred = cv2.imread(fold_pred + '/' + f)
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_gt = cv2.imread(fold_gt + '/' + f)
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_pred_gray = cv2.imread(fold_pred + '/' + f, 0).astype(np.float32) / 255.0
        img_gt_gray = cv2.imread(fold_gt + '/' + f, 0).astype(np.float32) / 255.0

        # # Convert images to tensors and move them to the GPU
        # img_pred_rgb_tensor = torch.from_numpy(img_pred_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        # img_gt_rgb_tensor = torch.from_numpy(img_gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        # img_pred_gray_tensor = torch.from_numpy(img_pred_gray).unsqueeze(0).unsqueeze(0).to(device)
        # img_gt_gray_tensor = torch.from_numpy(img_gt_gray).unsqueeze(0).unsqueeze(0).to(device)

        # Perform computations on the GPU
        PSNR_score = PSNR(img_gt_rgb, img_pred_rgb)
        SSIM_score = SSIM(img_gt_rgb, img_pred_rgb)
        L1_score = L1difference(img_gt_gray, img_pred_gray)
        # dist = loss_fn.forward(img_pred_rgb_tensor, img_gt_rgb_tensor)
        print(PSNR_score)

        PSNR_score_avg = PSNR_score_avg + PSNR_score
        SSIM_score_avg = SSIM_score_avg + SSIM_score
        L1_score_avg = L1_score_avg + L1_score
        # dist_avg = dist_avg + dist

    PSNR_score_avg = PSNR_score_avg / len(file_list)
    SSIM_score_avg = SSIM_score_avg / len(file_list)
    L1_score_avg = L1_score_avg / len(file_list)
    # dist_avg = dist_avg / len(file_list)

    print('SSIM: ', SSIM_score_avg)
    print('PSNR: ', PSNR_score_avg)
    print('L1: ', L1_score_avg)
    # print('lpips', dist_avg)
