import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils

from skimage import io, color
from skimage.filters import roberts, sobel_h, sobel_v
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from os import listdir
from os.path import isfile, join
import re
import cv2
import lpips

def PSNR(true_frame, pred_frame):
    eps = 0.0001
    prediction_error = 0.0
    dev_frame = pred_frame - true_frame
    dev_frame = np.multiply(dev_frame, dev_frame)
    prediction_error = np.mean(dev_frame)
    if prediction_error > eps:
        prediction_error = 10*np.log((1*1)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((1*1)/ eps)/np.log(10)
    return prediction_error


def SSIM(img1, img2):
    [h,w,c] = img1.shape
    if c > 2:
        img1 = color.rgb2yuv(img1)
        img1 = img1[:,:,0]
        img2 = color.rgb2yuv(img2)
        img2 = img2[:,:,0]
    score = ssim(img1, img2, data_range=1.0)
    return score


def L1difference(img_true,img_pred):
    [h,w] = img_true.shape
    true_gx = sobel_h(img_true)/4.0
    true_gy = sobel_v(img_true)/4.0
    pred_gx = sobel_h(img_pred)/4.0
    pred_gy = sobel_v(img_pred)/4.0
    dx = np.abs(true_gx-pred_gx)
    dy = np.abs(true_gy-pred_gy)
    prediction_error = np.mean(dx+dy)
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10*np.log((1*1)/ prediction_error)/np.log(10)
    else:
        prediction_error = 10*np.log((1*1)/ eps)/np.log(10)
    return prediction_error


if __name__ == "__main__":
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 3:
        print(f"Usage: python {script_name} [result_dir] [cfg_ratio]")
        sys.exit(0)

    result_dir = sys.argv[1].strip()
    cfg_ratio = float(sys.argv[2].strip())
    gt_suffix = '_reconstruction.png'
    pred_suffix = f'_samples_cfg_scale_{cfg_ratio:.2f}.png'

    #get images
    PSNR_score_avg = 0
    SSIM_score_avg = 0
    L1_score_avg = 0
    dist_avg = 0
    loss_fn = lpips.LPIPS(net='alex')
    gt_file_list = [f for f in sorted(os.listdir(result_dir)) if f[-len(gt_suffix):] == gt_suffix]
    pred_file_list = [f for f in sorted(os.listdir(result_dir)) if f[-len(pred_suffix):] == pred_suffix]
    assert len(gt_file_list) == len(pred_file_list)
    print(f'Totally {len(gt_file_list)} images to eval.')
    for (gt_file, pred_file) in zip(gt_file_list, pred_file_list):
        if gt_file[:-len(gt_suffix)] != pred_file[:-len(pred_suffix)]:
            print(gt_file, pred_file)
            continue

        img_pred = cv2.imread(result_dir+'/'+pred_file)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_gt = cv2.imread(result_dir+'/'+gt_file)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_pred_gray = cv2.imread(result_dir+'/'+pred_file,0).astype(np.float32)/255.0
        img_gt_gray = cv2.imread(result_dir+'/'+gt_file,0).astype(np.float32)/255.0


        pred_lpips = (img_pred - 0.5) / 0.5
        gt_lpips =  (img_gt - 0.5) / 0.5
        pred_tensor = torch.from_numpy(pred_lpips).permute(2, 0, 1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_lpips).permute(2, 0, 1).unsqueeze(0)

        PSNR_score = PSNR(img_gt, img_pred)
        SSIM_score = SSIM(img_gt, img_pred)
        L1_score = L1difference(img_gt_gray, img_pred_gray)
        dist = loss_fn.forward(pred_tensor, gt_tensor)

        print(PSNR_score,SSIM_score,L1_score,dist)
        PSNR_score_avg = PSNR_score_avg + PSNR_score
        SSIM_score_avg = SSIM_score_avg + SSIM_score
        L1_score_avg = L1_score_avg + L1_score
        dist_avg = dist_avg + dist
    PSNR_score_avg = PSNR_score_avg/len(gt_file_list)
    SSIM_score_avg = SSIM_score_avg/len(gt_file_list)
    L1_score_avg = L1_score_avg/len(gt_file_list)
    dist_avg  = dist_avg /len(gt_file_list)
    print('SSIM: ', SSIM_score_avg)
    print('PSNR: ', PSNR_score_avg)
    print('L1: ', L1_score_avg)
    print('lpips', dist_avg)
# srun -p bigdata  --cpus-per-task=92
# srun -p bigdata --gres=gpu:1 --cpus-per-task=32  python -m pytorch_fid /mnt/petrelfs/chenyuankun/ControlNet/results/202310211809/image_log/test_neg_pasd_plus/real /mnt/petrelfs/chenyuankun/ControlNet/results/202310211809/image_log/test_neg_pasd_plus/fake