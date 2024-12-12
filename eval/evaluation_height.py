import os
import sys
import cv2
import math
import numpy as np
from skimage import io


if __name__ == "__main__":
    max_height_value = 377.58
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 3:
        print(f"Usage: python {script_name} [result_dir] [gt_dir]")
        sys.exit(0)

    pred_suffix = '_samples_cfg_scale_9.00.png'
    result_dir = sys.argv[1].strip()
    gt_dir = sys.argv[2].strip()

    gt_file_list = {f.split('_')[0] : f for f in os.listdir(gt_dir)}
    pred_file_list = [f for f in sorted(os.listdir(result_dir)) if f[-len(pred_suffix):] == pred_suffix]
    print(f'Totally {len(pred_file_list)} predictions.')
    mse, mae = 0.0, 0.0
    pred_build_value, pred_ground_value = 0.0, 0.0
    for pred_file in pred_file_list:
        prefix = pred_file.split('_')[0]
        pred_file_path = os.path.join(result_dir, pred_file)
        gt_file_path = os.path.join(gt_dir, gt_file_list[prefix])

        # 30 is calculated by test data
        # 31.86 is calculated by train data
        gt_height_img = io.imread(gt_file_path) * 0.3048
        pred_height_img = (cv2.imread(pred_file_path, cv2.IMREAD_GRAYSCALE) / 255.0 * max_height_value * 0.3048) - 31.86

        build_mask = gt_height_img > 0.
        ground_mask = gt_height_img <= 0.
        pred_build_value += pred_height_img[build_mask].mean()
        pred_ground_value += pred_height_img[ground_mask].mean()

        diff = np.abs(gt_height_img[build_mask] - pred_height_img[build_mask])
        mse += np.power(diff, 2).mean()
        mae += diff.mean()

    mse /= len(pred_file_list)
    mae /= len(pred_file_list)
    rmse = math.sqrt(mse)
    print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')

    pred_build_value /= len(pred_file_list)
    pred_ground_value /= len(pred_file_list)
    print(f'Pred ground value: {pred_ground_value}, Pred build value: {pred_build_value}')


# python eval/evaluation_height.py \
#     lightning_logs/version_1549720/results/ \
#     /mnt/petrelfs/share_data/bigdata_rs/datasets/OmniCity/OmniCity/OmniCity-dataset-trainval/satellite-level/annotation-height/annotation-height-test