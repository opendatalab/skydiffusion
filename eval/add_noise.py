import cv2
import numpy as np

# 读取图像
width = 1024
height = 512

# 生成高斯噪声图像
noise = np.random.normal(128, 64, (height, width)).astype(np.uint8)  # 均值为128，标准差为64

# 创建一个全黑图像
image = np.zeros((height, width), dtype=np.uint8)

# 将噪声叠加到图像上
image += noise
# 保存带有噪声的图像
cv2.imwrite('/mnt/petrelfs/chenyuankun/ControlNet/eval/noisy_image.png', image)
