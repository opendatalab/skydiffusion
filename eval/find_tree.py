import os
import shutil

train_csv = '/mnt/petrelfs/chenyuankun/ControlNet/cleaned_sky_test.csv'
train_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/street-level/rotate-panorama/rotate-panorama-test'
save_data = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/street-level/rotate-panorama/rotate-panorama-tree' 

cnt = 0
sourcelist = []
with open(train_csv, 'rt') as f:
    for line in f:
        segs = line.strip().split('\t')
        temp = os.path.basename(segs[2].strip())
        # dir_name, base_name = os.path.split(segs[2].strip())
        sourcelist.append(temp)
        # target_path = os.path.join(save_data, base_name)
        # if not os.path.exists(os.path.join(train_dir, base_name)):
        #     cnt += 1
        #     shutil.copy(source_path, target_path)
trainlist = os.listdir(train_dir)
for f in trainlist:
    if f not in sourcelist:
        cnt += 1
        # name = os.path.basename(f)
        shutil.copy(os.path.join(train_dir,f), os.path.join(save_data, f))

print(cnt)