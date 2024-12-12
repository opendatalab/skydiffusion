import os
import shutil

# 源目录
source_directory = "/mnt/petrelfs/chenyuankun/ControlNet/results/geo+atten/image_log/test_neg2"

# 目标目录
real_directory = "/mnt/petrelfs/chenyuankun/ControlNet/results/geo+atten/image_log/test_neg2/real"
fake_directory = "/mnt/petrelfs/chenyuankun/ControlNet/results/geo+atten/image_log/test_neg2/fake"
if not os.path.exists(real_directory):
    os.mkdir(real_directory)
if not os.path.exists(fake_directory):
    os.mkdir(fake_directory)

# 获取源目录下所有文件
for filename in os.listdir(source_directory):
    if filename.endswith("_reconstruction.png"):
        # 构建完整的源文件路径
        source_file = os.path.join(source_directory, filename)
        # 构建完整的目标文件路径
        target_file = os.path.join(real_directory, filename.replace("_reconstruction.png", ".png"))
        # 移动文件
        shutil.move(source_file, target_file)
    elif filename.endswith("_samples_cfg_scale_9.00.png"):
        # 构建完整的源文件路径
        source_file = os.path.join(source_directory, filename)
        # 构建完整的目标文件路径
        target_file = os.path.join(fake_directory, filename.replace("_samples_cfg_scale_9.00.png", ".png"))
        # 移动文件
        shutil.move(source_file, target_file)
