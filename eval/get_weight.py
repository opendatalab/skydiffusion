import os
import sys
import cv2
import math
import numpy as np
import os.path as osp
from skimage import io
from scipy import sparse
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import transforms


def generate_grid(h, w):
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xy = np.meshgrid(x, y)

    grid = torch.tensor([xy]).float()
    grid = grid.permute(0, 2, 3, 1)
    return grid


def depth2voxel(img_depth, gsize):
    gsize = torch.tensor(gsize).int()
    n, c, h, w = img_depth.size()
    # meter
    # site_z = img_depth[:, 0, int(h/2), int(w/2)] + 2.5
    # feet
    site_z = img_depth[:, 0, int(h/2), int(w/2)] + 8.2
    voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize)

    # depth voxel
    grid_mask = generate_grid(gsize, gsize)
    grid_mask = grid_mask.expand(n, gsize, gsize, 2)
    grid_depth = torch.nn.functional.grid_sample(img_depth, grid_mask, mode='nearest', align_corners=True)
    voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
    voxel_depth = voxel_depth - voxel_sitez

    # occupancy voxel
    voxel_grid = torch.arange(-gsize/2, gsize/2, 1).float()
    voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize)
    voxel_ocupy = torch.ge(voxel_depth, voxel_grid).float().cpu()
    voxel_ocupy[:,gsize-1,:,:] = 0
    voxel_ocupy = voxel_ocupy

    # distance voxel
    voxel_dx = grid_mask[0,:,:,0].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dy = grid_mask[0,:,:,1].view(1,1,gsize,gsize).expand(n,gsize,gsize,gsize).float()*float(gsize/2.0)
    voxel_dz = voxel_grid

    voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(voxel_dz)
    voxel_dis = voxel_dis.add(0.01)   # avoid 1/0 = nan
    voxel_dis = voxel_dis.mul(voxel_ocupy)
    voxel_dis = torch.sqrt(voxel_dis) - voxel_ocupy.add(-1.0).mul(float(gsize)*0.9)
    return voxel_dis


def voxel2pano(voxel_dis, size_pano, ori=torch.Tensor([0])):
    PI = 3.1415926535
    r, c = [size_pano[0], size_pano[1]]
    n, s, t, tt = voxel_dis.size()
    k = int(s/2)

    # rays
    ori = ori.view(n, 1).expand(n, c).float()
    x = torch.arange(0, c, 1).float().view(1, c).expand(n, c)
    y = torch.arange(0, r, 1).float().view(1, r).expand(n, r)
    lon = x * 2 * PI/c + ori - PI
    lat = PI/2.0 - y * PI/r
    sin_lat = torch.sin(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
    sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
    vx =  cos_lat.mul(sin_lon)
    vy = -cos_lat.mul(cos_lon)
    vz =  sin_lat
    vx = vx.expand(n, k, r, c)
    vy = vy.expand(n, k, r, c)
    vz = vz.expand(n, k, r, c)

    #
    voxel_dis = voxel_dis.contiguous().view(1, n*s*s*s)
    idx_x = (np.linspace(-1.0, 1.0, s) + 1) / 2 * (r-1)
    idx_y = (np.linspace(-1.0, 1.0, s) + 1) / 2 * (r-1)
    idx_xx, idx_yy = np.meshgrid(np.rint(idx_x), np.rint(idx_y))
    voxel_idx = r * idx_yy + idx_xx
    voxel_idx = torch.tensor(voxel_idx).float().view(1, 1, s, s).expand(n, s, s, s)
    voxel_idx = voxel_idx.contiguous().view(1, n*s*s*s)

    # sample voxels along pano-rays
    d_samples = torch.arange(0, float(k), 1).view(1, k, 1, 1).expand(n, k, r, c)
    samples_x = vx.mul(d_samples).add(k).long()
    samples_y = vy.mul(d_samples).add(k).long()
    samples_z = vz.mul(d_samples).add(k).long()
    samples_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, k, r, c).long()
    samples_indices = samples_n.mul(s*s*s).add(samples_z.mul(s*s)).add(samples_y.mul(s)).add(samples_x)
    samples_indices = samples_indices.view(1, n*k*r*c)
    samples_indices = samples_indices[0,:]

    # get depth pano
    samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
    samples_depth = samples_depth.view(n, k, r, c)
    min_depth = torch.min(samples_depth, 1)
    pano_depth = min_depth[0]
    pano_depth = pano_depth.view(n, 1, r, c)

    # get satellite image pixel index
    idx_z = min_depth[1].cpu().long()
    idx_y = torch.arange(0, r, 1).view(1, r, 1).expand(n, r, c).long()
    idx_x = torch.arange(0, c, 1).view(1, 1, c).expand(n, r, c).long()
    idx_n = torch.arange(0, n, 1).view(n, 1, 1).expand(n, r, c).long()
    idx = idx_n.mul(k*r*c).add(idx_z.mul(r*c)).add(idx_y.mul(c)).add(idx_x).view(1, n*r*c)
    samples_i = torch.index_select(voxel_idx, 1, samples_indices)
    pano_index = torch.index_select(samples_i, 1, idx[0,:]).view(n,1,r,c).float()
    return pano_depth, pano_index


def geo_projection(sate_depth, ori, sate_gsd=0.5, pano_size=[512, 1024]):
    gsize = sate_depth.size()[3] * sate_gsd
    voxel_dis = depth2voxel(sate_depth, gsize)
    ori = torch.Tensor([ori/180.0 * 3.1415926535])
    pano_depth, pano_index = voxel2pano(voxel_dis, pano_size, ori)
    pano_depth = pano_depth.mul(1.0 / (0.9 * gsize))
    return pano_index


def calc_distance(row1, col1, row2, col2):
    return math.sqrt(math.pow(row1-row2, 2.0) + math.pow(col1-col2, 2.0))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_func(distance):
    score = 1.0 - sigmoid(0.5*distance)
    return score if score >= 1e-2 else 0


def generate_weight_map(pano_size, tgt_row, tgt_col):
    (pano_height, pano_width) = pano_size
    result_map = np.zeros(pano_size)
    for row in range(pano_height):
        for col in range(pano_width):
            distance = calc_distance(row, col, tgt_row, tgt_col)
            result_map[row, col] = score_func(distance)
    return result_map


def process_func(item):
    height_img_path, output_path = item
    if osp.exists(output_path):
        return

    # geo transformation
    height_img = io.imread(height_img_path)
    height_img_tensor = transforms.ToTensor()(height_img[..., None])
    height_img_tensor = torch.stack([height_img_tensor])

    # omnicity
    # ori = 180.0
    # pano_size = (512, 1024)
    # cvusa
    ori = 0.0
    pano_size = (256, 1024)

    # feet
    sate_gsd = 0.859
    pano_index = geo_projection(height_img_tensor, ori, sate_gsd, pano_size).permute(0, 2, 3, 1)[0].data.cpu().numpy()

    # omnicity
    # pano_width, pano_height = 1024, 512
    # cvusa
    pano_width, pano_height = 1024, 256

    resized_pano_width = pano_width // 8
    resized_pano_height = pano_height // 8

    # omnicity
    # sate_height, sate_width = 512, 512
    # cvusa
    sate_height, sate_width = 256, 256

    resized_sate_width = sate_width // 8
    resized_sate_height = sate_height // 8

    pano_pixel_num = resized_pano_height * resized_pano_width
    sate_pixel_num = resized_sate_height * resized_sate_width
    full_weight_map = np.zeros((pano_pixel_num, sate_pixel_num))
    for src_row in range(resized_pano_height):
        for src_col in range(resized_pano_width):
            sate_idx = pano_index[src_row*8][src_col*8]
            sate_row = sate_idx // sate_width
            sate_col = sate_idx % sate_width
            tgt_row = sate_row // 8
            tgt_col = sate_col // 8
            weight_index = src_row * resized_pano_width + src_col
            weight_map = generate_weight_map((resized_sate_height, resized_sate_width), tgt_row, tgt_col)
            full_weight_map[weight_index, :] = weight_map.reshape(-1)

    full_weight_map=sparse.csr_matrix(full_weight_map)
    sparse.save_npz(output_path,full_weight_map)


def main(input_file, output_dir, output_file):
    # train
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view1/view1-train'
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view2/view2-train'
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view3/view3-train'
    # height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-height/annotation-height-train'
    # test
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view1/view1-test'
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view2/view2-test'
    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/image-satellite/satellite-image-view3/view3-test'
    # height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/ominicity_origin/satellite-level/annotation-height/annotation-height-test'

    # sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/satellite-level/image-satellite/'
    # height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/omnicity/satellite-level/annotation-height/'

    # cvusa
    sate_img_dir = '/mnt/petrelfs/share_data/chenyuankun/cvusa/bingmap'
    # train
    height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/cvusa/cvusa_height_train'
    # test
    # height_img_dir = '/mnt/petrelfs/share_data/chenyuankun/cvusa/cvusa_height_test'

    to_process_items = []
    writer = open(output_file, 'w')
    for line in open(input_file, 'r').readlines():
        segs = line.strip().split('\t')
        img_name = osp.basename(segs[1].strip())[:-4]
        sate_img_path = osp.join(sate_img_dir, img_name + '.jpg')
        height_img_path = osp.join(height_img_dir, img_name + '.tif')
        if not osp.exists(sate_img_path) or not osp.exists(height_img_path):
            print(f"{sate_img_path} or {height_img_path} not existed.")
            continue
        to_process_items.append((height_img_path, osp.join(output_dir, f"{img_name}.npz")))
        segs.append(sate_img_path)
        segs.append('0')
        segs.append(osp.join(output_dir, f"{img_name}.npz"))
        writer.write('\t'.join(segs) + '\n')
    writer.flush()
    writer.close()

    print(f"Totally {len(to_process_items)} images to process.")
    with Pool(32) as p:
        p.map(process_func, to_process_items)


if __name__ == '__main__':
    script_name = sys.argv[0].strip()
    if len(sys.argv) != 4:
        print(f"Usage: python {script_name} [input_file] [output_dir] [output_file]")
        sys.exit(0)

    input_file = sys.argv[1].strip()
    output_dir = sys.argv[2].strip()
    output_file = sys.argv[3].strip()
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    main(input_file, output_dir, output_file)


# python data/omnicity/sate_attn_preprocess.py \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_trainval.csv \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/sate-image-weight/train \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_sate_attn_trainval.csv

# python data/omnicity/sate_attn_preprocess.py \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_test.csv \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/sate-image-weight/test \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/geo-height_sate_attn_test.csv

# python data/omnicity/sate_attn_preprocess.py \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-train.csv \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/sate-image-weight \
#     /mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height-sate_attn/random-cleaned2-train.csv



# cvusa
# python data/omnicity/sate_attn_preprocess.py /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/cdte/train.csv /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate-image-weight /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/cdte-train.csv
# python data/omnicity/sate_attn_preprocess.py /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/cdte/val.csv /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate-image-weight /mnt/petrelfs/share_data/zhonghuaping.p/datasets/CVUSA/sate_attn/cdte-val.csv