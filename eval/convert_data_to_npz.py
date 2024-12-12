import os
import sys
import cv2
import numpy as np


def main(result_dir, out_dir):
    gt_suffix = '_reconstruction.png'
    pred_suffix = '_samples_cfg_scale_9.00.png'

    image_names = sorted([f for f in os.listdir(result_dir) if f[-len(gt_suffix):] == gt_suffix])
    gt_images = [[cv2.imread(os.path.join(result_dir, name))] for name in image_names]
    gt_images = np.concatenate(gt_images, axis=0)
    # print(gt_images.shape)
    np.savez(os.path.join(out_dir, 'gt.npz'), gt_images)

    image_names = sorted([f for f in os.listdir(result_dir) if f[-len(pred_suffix):] == pred_suffix])
    pred_images = [[cv2.imread(os.path.join(result_dir, name))] for name in image_names]
    pred_images = np.concatenate(pred_images, axis=0)
    # print(pred_images.shape)
    np.savez(os.path.join(out_dir, 'pred.npz'), pred_images)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: [result_dir] [out_dir]')
        sys.exit(0)

    result_dir = sys.argv[1].strip()
    out_dir = sys.argv[2].strip()
    main(result_dir, out_dir)