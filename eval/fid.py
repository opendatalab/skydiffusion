"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import pytorch_fid
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np
import pytorch_fid
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    # mu_matrix = np.matrix(mu)
    # cov_matrix = np.matrix(cov)
    # mu2_matrix = np.matrix(mu2)
    # cov2_matrix = np.matrix(cov2)
    # print(mu,mu2,cov,cov2)
    # print(mu)
    # print(mu2)
    cc, _ = linalg.sqrtm(np.matrix(np.dot(cov, cov2)), disp=False)
    # print(f'cc: {cc}')
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    # print(dist)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=1):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    path1 = '/mnt/petrelfs/chenyuankun/comingdowntoearth/test_demo/streets'
    path2 = '/mnt/petrelfs/chenyuankun/comingdowntoearth/test_demo/fake_streets'
    path3 = '/mnt/petrelfs/chenyuankun/comingdowntoearth/eval/cvusa/cdte/fake'
    path4 = '/mnt/petrelfs/chenyuankun/comingdowntoearth/eval/cvusa/cdte/real'


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize([112, 616]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        # transforms.Normalize(mean=mean, std=std)
    ])

    filenames1 = sorted([f for f in os.listdir(path1)])
    filenames2 = sorted([f for f in os.listdir(path2)])
    filenames3 = sorted([f for f in os.listdir(path3)])
    filenames4 = sorted([f for f in os.listdir(path4)])
    print(path1,path2,path3,path4)
    # print(filenames1,filenames2)
    # actvs1 = []
    # actvs2 = []
    cnt = 0
    for x in range(len(filenames1)):
        # print(x)
        file1 = filenames1[x]
        file1 = os.path.join(path1,file1)
        img1 = Image.open(file1).convert('RGB')
        img1 = transform(img1).unsqueeze(0)
        file2 = filenames2[x]
        file2 = os.path.join(path2,file2)
        img2 = Image.open(file2).convert('RGB')
        img2 = transform(img2).unsqueeze(0)
        actvs1 = []
        actvs2 = []
        actvs1.append(inception(img1.to(device)))
        actvs1_all = torch.cat(actvs1, dim=0).cpu().detach().numpy()
        actvs2.append(inception(img2.to(device)))
        actvs2_all = torch.cat(actvs2, dim=0).cpu().detach().numpy()
        mu, cov = [], []
        mu.append(np.mean(actvs1_all, axis=0))
        cov.append(np.cov(actvs1_all, rowvar=False))
        mu.append(np.mean(actvs2_all, axis=0))
        cov.append(np.cov(actvs2_all, rowvar=False))
        fid_value1 = frechet_distance(mu[0], cov[0], mu[1], cov[1])
        


        file3 = filenames3[x]
        file3 = os.path.join(path3,file3)
        img3 = Image.open(file3).convert('RGB')
        img3 = transform(img3).unsqueeze(0)
        file4 = filenames4[x]
        file4 = os.path.join(path4,file4)
        img4 = Image.open(file4).convert('RGB')
        img4 = transform(img4).unsqueeze(0)
        actvs1 = []
        actvs2 = []
        actvs1.append(inception(img3.to(device)))
        actvs1_all = torch.cat(actvs1, dim=0).cpu().detach().numpy()
        actvs2.append(inception(img4.to(device)))
        actvs2_all = torch.cat(actvs2, dim=0).cpu().detach().numpy()
        mu, cov = [], []
        mu.append(np.mean(actvs1_all, axis=0))
        cov.append(np.cov(actvs1_all, rowvar=False))
        mu.append(np.mean(actvs2_all, axis=0))
        cov.append(np.cov(actvs2_all, rowvar=False))
        fid_value2 = frechet_distance(mu[0], cov[0], mu[1], cov[1])
        

        if(fid_value2 > fid_value1 + 300) :
            print(f'value1: {fid_value1}')
            print(f'value2: {fid_value2}')
            print(file1,file2,file3,file4)
            cnt +=1
    print(cnt)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', default='/mnt/petrelfs/chenyuankun/comingdowntoearth/test_demo')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size to use')
    args = parser.parse_args()
    paths = []
    paths.append('/mnt/petrelfs/chenyuankun/comingdowntoearth/test_demo/streets')
    paths.append('/mnt/petrelfs/chenyuankun/comingdowntoearth/test_demo/fake_streets')
    fid_value = calculate_fid_given_paths(paths, args.img_size, args.batch_size)
    print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE