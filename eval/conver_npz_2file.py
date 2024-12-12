import os
import sys
import cv2
import numpy as np


def main(gt_dir, pred_dir, out_dir):
    gt_suffix = '_reconstruction.png'
    pred_suffix = '_samples_cfg_scale_9.00.png'

    # image_names = sorted([f for f in os.listdir(result_dir) if f[-len(gt_suffix):] == gt_suffix])
    image_names = sorted([f for f in os.listdir(gt_dir)])
    gt_images = [[cv2.imread(os.path.join(gt_dir, name))] for name in image_names]
    gt_images = np.concatenate(gt_images, axis=0)
    # print(gt_images.shape)
    np.savez(os.path.join(out_dir, 'gt.npz'), gt_images)

    # image_names = sorted([f for f in os.listdir(result_dir) if f[-len(pred_suffix):] == pred_suffix])
    image_names = sorted([f for f in os.listdir(pred_dir)])
    pred_images = [[cv2.imread(os.path.join(pred_dir, name))] for name in image_names]
    pred_images = np.concatenate(pred_images, axis=0)
    # print(pred_images.shape)
    np.savez(os.path.join(out_dir, 'pred.npz'), pred_images)


if __name__ == '__main__':
    gt_dir ='/mnt/petrelfs/chenyuankun/sate_to_ground/bicycle_net/results_origin/streets'
    pred_dir = '/mnt/petrelfs/chenyuankun/sate_to_ground/bicycle_net/results_origin/fake_streets'
    out_dir = '/mnt/petrelfs/chenyuankun/sate_to_ground/bicycle_net/results_origin/npz'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main(gt_dir,pred_dir, out_dir)