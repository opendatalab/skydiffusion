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
# from skimage.metrics import structural_similarity as ssim
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
def PSNR(true_frame, pred_frame):
    eps = 0.0001
    prediction_error = 0.0
    [h,w,c] = true_frame.shape
    dev_frame = pred_frame-true_frame
    dev_frame = np.multiply(dev_frame,dev_frame)
    # prediction_error = np.sum(dev_frame)
    # prediction_error = 128*128*prediction_error/(h*w*c)
    prediction_error = np.mean(dev_frame)
    if prediction_error > eps:
        # prediction_error = 10*np.log((255*255)/ prediction_error)/np.log(10)
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
    score = ssim(img1, img2,data_range=1.0)
    # score = ssim(img1, img2, data_range=1.0, multichannel=False)
    return score

def SSIM_RGB(img1, img2):
    score = ssim(img1, img2, data_range=1.0, multichannel=True)
    return score
def L1difference(img_true,img_pred):
    [h,w] = img_true.shape
    true_gx = sobel_h(img_true)/4.0
    true_gy = sobel_v(img_true)/4.0 
    pred_gx = sobel_h(img_pred)/4.0
    pred_gy = sobel_v(img_pred)/4.0
    dx = np.abs(true_gx-pred_gx)
    dy = np.abs(true_gy-pred_gy)
    # prediction_error = np.sum(dx+dy)
    # prediction_error=128*128*prediction_error/(h*w)
    prediction_error = np.mean(dx+dy)
    eps = 0.0001
    if prediction_error > eps:
        prediction_error = 10*np.log((1*1)/ prediction_error)/np.log(10)
    else:         
        prediction_error = 10*np.log((1*1)/ eps)/np.log(10)
    return prediction_error





if __name__ == "__main__":
    # script_name = sys.argv[0].strip()
    # if len(sys.argv) != 3:
    #     print(f"Usage: python {script_name} [pred_dir] [gt_dir]")
    #     sys.exit(0)

    # fold_pred = '/mnt/petrelfs/chenyuankun/ControlNet/results/202310211809_geo512/image_log/test_neg_p5/fake'
    # fold_gt = '/mnt/petrelfs/chenyuankun/ControlNet/results/202310211809_geo512/image_log/test_neg_p5/real'
    # fold_pred = '/mnt/petrelfs/chenyuankun/comingdowntoearth/eval/SD/fake'
    # fold_gt = '/mnt/petrelfs/chenyuankun/comingdowntoearth/eval/SD/real'
    fold_pred = sys.argv[1].strip()
    fold_gt = sys.argv[2].strip()
    #get images
    PSNR_score_avg = 0
    SSIM_score_avg = 0
    L1_score_avg = 0
    dist_avg = 0
    file_list = [f for f in os.listdir(fold_pred) if os.path.isfile(os.path.join(fold_pred,f))]
    # loss_fn = lpips.LPIPS(net='alex')


    width, height = 1024, 512
    i = 0
    # fold_gt = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/rotated-image-panorama'
    for f in file_list:
        i += 1
        img_pred = cv2.imread(fold_pred+'/'+f)
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_gt = cv2.imread(fold_gt+'/'+f)
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        
        # img_pred = cv2.resize(img_pred, (width, height))
        # img_gt = cv2.resize(img_gt, (width, height))

        img_pred_gray = cv2.imread(fold_pred+'/'+f,0).astype(np.float32)/255.0
        img_gt_gray = cv2.imread(fold_gt+'/'+f,0).astype(np.float32)/255.0


        # img_pred_gray = cv2.resize(img_pred_gray, (width, height))
        # img_gt_gray = cv2.resize(img_gt_gray, (width, height))
        pred_lpips = (img_pred_rgb - 0.5) / 0.5
        gt_lpips =  (img_gt_rgb - 0.5) / 0.5
        pred_tensor = torch.from_numpy(pred_lpips).permute(2, 0, 1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_lpips).permute(2, 0, 1).unsqueeze(0)
        

        
        PSNR_score = PSNR(img_gt_rgb, img_pred_rgb)
        SSIM_score = SSIM(img_gt_rgb, img_pred_rgb)
        L1_score = L1difference(img_gt_gray, img_pred_gray)
        # dist = loss_fn.forward(pred_tensor, gt_tensor)


        # print(PSNR_score,SSIM_score,L1_score,dist)
        # print(PSNR_score,dist)

        PSNR_score_avg = PSNR_score_avg + PSNR_score
        SSIM_score_avg = SSIM_score_avg + SSIM_score
        L1_score_avg = L1_score_avg + L1_score
        # dist_avg = dist_avg + dist

    PSNR_score_avg = PSNR_score_avg/len(file_list)
    SSIM_score_avg = SSIM_score_avg/len(file_list)
    L1_score_avg = L1_score_avg/len(file_list)
    # dist_avg  = dist_avg /len(file_list)
    print('SSIM: ', SSIM_score_avg)
    print('PSNR: ', PSNR_score_avg)
    print('L1: ', L1_score_avg)
    # print('lpips', dist_avg)

