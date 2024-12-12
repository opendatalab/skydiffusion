import os

# 读取两个文件名列表
# train_file_list = '/mnt/petrelfs/chenyuankun/Sat2Density/train.txt'
test_file_list = '/mnt/petrelfs/share_data/chenyuankun/omnicity/random_test_cleaned2.txt'

# 两个文件夹路径
folder1 = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/sky-mask'
folder2 = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/rotated-image-panorama'
# folder3 = '/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/geo-height-panorama/test'
# folder3 = '/mnt/petrelfs/share_data/chenyuankun/omnicity/street-level/geo-height-panorama'
# train_csv = '/mnt/petrelfs/chenyuankun/Sat2Density/new_density_train.csv'
test_csv = '/mnt/petrelfs/chenyuankun/ControlNet/extra_skymask.csv'

# 从文件名列表中读取文件名
def read_file_list(file_path):
    with open(file_path, 'r') as file:
        file_list = file.read().splitlines()
    return file_list


# 生成CSV文件
def generate_csv(file_list, folder_path, output_csv):
    with open(output_csv, 'w') as csv_file:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            # 根据你的需求构建CSV行
            name = file_name.split('_')[0]
            # csv_line = f"{file_path}\t{file_path.replace(folder_path, folder2)}\t{file_path.replace(folder_path, folder3)}"
            # csv_line = f"{name}\t{file_path.replace(folder_path, folder2)}\t{file_path.replace(folder_path, folder3)}"
            csv_line = f"{name}\t{file_path.replace(folder_path, folder1)}\t{file_path.replace(folder_path, folder2)}"
            csv_file.write(csv_line + '\n')

# 读取文件名列表并生成CSV
# train_file_list = read_file_list(train_file_list)
test_file_list = read_file_list(test_file_list)
# generate_csv(train_file_list, folder1, train_csv)
generate_csv(test_file_list, folder1, test_csv)

# print(f"{len(train_file_list)} files added to {train_csv}")
print(f"{len(test_file_list)} files added to {test_csv}")
# print('256')