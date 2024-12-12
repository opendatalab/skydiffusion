import os
import cv2
import random
import pickle
import numpy as np
from cruvedbev import CruvedBEV

from torch.utils.data import Dataset


class SkyDataset(Dataset):
    def __init__(self, data_file, is_cvusa, prompt, image_size=(512, 512), drop_prompt_ratio=0, dataset_name="CVACT"):
        self.data = []
        self.image_size = image_size
        self.drop_prompt_ratio = drop_prompt_ratio
        self.dataset_name = dataset_name
        with open(data_file, 'rt') as f:
            print(is_cvusa)
            if not is_cvusa:
                for line in f:
                    segs = line.strip().split('\t')
                    # condition_name = os.path.basename()
                    dir_name, base_name = os.path.split(segs[1].strip())
                    base_name = base_name.split('_')[0]+'.png'
                    condition = os.path.join(dir_name,base_name)
                    if len(segs) == 4:
                        self.data.append({
                            'name': segs[0].strip(),
                            'source': segs[1].strip(),
                            'target': segs[2].strip(),
                            'prompt': segs[3].strip(),
                        })
                    else:
                        # print(segs[1].strip())
                        self.data.append({
                            'name': segs[0].strip(),
                            'source': segs[1].strip(),
                            # 'source': condition,
                            'target': segs[2].strip(),
                            'prompt': prompt.strip(),
                        })
            else:
                for line in f:
                    segs = line.strip().split(',')
                    basename = os.path.basename(segs[0].strip())
                    file_name, ext = os.path.splitext(basename)
                    if len(segs) == 4:
                        self.data.append({
                            'name': file_name,
                            'source': segs[1].strip(),
                            'target': segs[2].strip(),
                            'prompt': segs[3].strip(),
                        })
                    else:
                        self.data.append({
                            'name': file_name,
                            'source': segs[2].strip(),
                            'target': segs[1].strip(),
                            'prompt': prompt.strip(),
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data)
        item = self.data[idx]
        # print('item',item)
        name = item['name']
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        # print(source_filename)
        # randomly drop text prompt
        if self.drop_prompt_ratio > 0 and random.random() < self.drop_prompt_ratio:
            prompt = ""

        if os.path.splitext(source_filename)[1] == '.pkl':
            source = pickle.load(open(source_filename, 'rb'))
        else:
            source = CruvedBEV(source_filename, self.dataset_name)
            
            # print(source,source_filename)
            # Do not forget that OpenCV read images in BGR order.
            # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

            # try:
            #     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            # except Exception as e:
            #     print(f"An error occurred: {e}")
            #     print(source)



        
        source = cv2.resize(source, self.image_size)

        target = cv2.imread(target_filename)
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, self.image_size)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, name=name)

