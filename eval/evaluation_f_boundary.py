"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code adapted from:
# https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/f_boundary.py
#
# Source License
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.s
##############################################################################
#
# Based on:
# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------
"""


import os
import cv2

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from skimage.morphology import binary_dilation,disk

""" Utilities for computing, reading and saving benchmark evaluation."""


def eval_mask_boundary(seg_mask,gt_mask,num_classes,bound_th=0.008):
	"""
	Compute F score for a segmentation mask

	Arguments:
		seg_mask (ndarray): segmentation mask prediction
		gt_mask (ndarray): segmentation mask ground truth
		num_classes (int): number of classes

	Returns:
		F (float): mean F score across all classes
		Fpc (listof float): F score per class
	"""
	Fc = 0.0
	count = 0
	for class_id in tqdm(range(num_classes)):
		seg_i = (seg_mask==class_id).astype(np.float)
		gt_i = (gt_mask==class_id).astype(np.float)
		if np.sum(seg_i)>100 and np.sum(gt_i)>100:
			Fc += db_eval_boundary(seg_i, gt_i)
			count += 1
	return Fc/count


def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.

	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask		 (ndarray): binary annotated image.

	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	#bound_pix = 1

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap2(foreground_mask)
	gt_boundary = seg2bmap2(gt_mask)

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg	 = np.sum(fg_boundary)
	n_gt	 = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall	= np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall)

	#return F, precision
	return F

def seg2bmap(seg,width=None,height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.

	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]

	Returns:
		bmap (ndarray):	Binary boundary map.

	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+floor((y-1)+height / h)
					i = 1+floor((x-1)+width  / h)
					bmap[j,i] = 1

	return bmap


def seg2bmap2(seg,width=None,height=None):
	bmap = cv2.Laplacian(seg,cv2.CV_64F)
	bmap[bmap>0] = 1
	bmap[bmap<0] = 1
	return bmap

def F_Sem():
	fold = "D:/permanent/aligned_2k/seg_compare"
	fold_gt = fold + '/gt/result'
	fold_p2p = fold + '/p2p/images/output/result'
	fold_cvpr = fold + '/cvpr2018/images/output/result'
	fold_our = fold + '/our/rgb/result'
	fold_abl_1 = fold + '/alb/L2R_good_with_depth_no_mask/result'
	fold_abl_2 = fold + '/alb/L2R_good_without_depth/result'
	fold_abl_3 = fold + '/alb/L2R_good_without_depth_geo/result'
	img_id = []	
	img_all = [f for f in os.listdir(fold_gt) if os.path.isfile(os.path.join(fold_gt,f))]
	for img_name in img_all:
		if img_name[-18:] == '_street_rgb_00.png': 
			name = img_name[:-18]				
			img_id.append(name)
	
	f_cvpr_avg = 0.0
	f_p2p_avg = 0.0
	f_our_avg = 0.0
	f_abl1_avg = 0.0
	f_abl2_avg = 0.0
	f_abl3_avg = 0.0
	for i in range(0,len(img_id)):
		print(img_id[i])
		img_gt = cv2.imread(fold_gt+'/'+img_id[i]+ '_street_rgb_00.png').astype(np.float)
		mask_gt = img_gt[:,:,0] + img_gt[:,:,1]+ img_gt[:,:,2]

		img_cvpr = cv2.imread(fold_cvpr+'/'+img_id[i]+ '_cvpr2018_00.png').astype(np.float)
		mask_cvpr = img_cvpr[:,:,0] + img_cvpr[:,:,1]+ img_cvpr[:,:,2]

		img_p2p = cv2.imread(fold_p2p+'/'+img_id[i]+ '_cvpr2018_00.png').astype(np.float)
		mask_p2p = img_p2p[:,:,0]+ img_p2p[:,:,1] + img_p2p[:,:,2]

		img_our = cv2.imread(fold_our+'/'+img_id[i]+ '.png').astype(np.float)
		mask_our = img_our[:,:,0]+ img_our[:,:,1] + img_our[:,:,2]

		img_abl1 = cv2.imread(fold_abl_1+'/'+img_id[i]+ '.png').astype(np.float)
		mask_abl1 = img_abl1[:,:,0] + img_abl1[:,:,1]+ img_abl1[:,:,2]

		img_abl2 = cv2.imread(fold_abl_2+'/'+img_id[i]+ '.png').astype(np.float)
		mask_abl2 = img_abl2[:,:,0] + img_abl2[:,:,1]+ img_abl2[:,:,2]

		img_abl3 = cv2.imread(fold_abl_3+'/'+img_id[i]+ '.png').astype(np.float)
		mask_abl3 = img_abl3[:,:,0] + img_abl3[:,:,1]+ img_abl3[:,:,2]
		
		# pix accuracy
		# db_eval_boundary(foreground_mask,gt_mask, ignore_mask,bound_th=0.008):
		if 0:
			f_cvpr_avg += eval_mask_boundary(mask_cvpr, mask_gt, 255*3)
			f_p2p_avg += eval_mask_boundary(mask_p2p, mask_gt, 255*3)
			f_our_avg += eval_mask_boundary(mask_our, mask_gt, 255*3)
			f_abl1_avg += eval_mask_boundary(mask_abl1, mask_gt, 255*3)
			f_abl2_avg += eval_mask_boundary(mask_abl2, mask_gt, 255*3)
			f_abl3_avg += eval_mask_boundary(mask_abl3, mask_gt, 255*3)
		else:						
			f_cvpr_avg += db_eval_boundary(mask_cvpr, mask_gt)
			f_p2p_avg += db_eval_boundary(mask_p2p, mask_gt)
			f_our_avg += db_eval_boundary(mask_our, mask_gt)
			f_abl1_avg += db_eval_boundary(mask_abl1, mask_gt)
			f_abl2_avg += db_eval_boundary(mask_abl2, mask_gt)
			f_abl3_avg += db_eval_boundary(mask_abl3, mask_gt)						
		
	print(f_cvpr_avg/len(img_id))
	print(f_p2p_avg/len(img_id))
	print(f_our_avg/len(img_id))
	print(f_abl1_avg/len(img_id))
	print(f_abl2_avg/len(img_id))
	print(f_abl3_avg/len(img_id))

def F_Depth():
	fold = "D:/permanent/aligned_2k/rebuttal/fscore/depth"
	fold_gt = fold + '/gt'
	fold_pred = fold + '/pred'
	img_id = []	
	img_all = [f for f in os.listdir(fold_gt) if os.path.isfile(os.path.join(fold_gt,f))]
	for img_name in img_all:
		if img_name[-18:] == '_proj_label_00.png': 
			name = img_name[:-18]				
			img_id.append(name)
	
	f_avg = 0.0
	for i in range(0,len(img_id)):
		print(img_id[i])
		img_gt = cv2.imread(fold_gt+'/'+img_id[i]+ '_proj_dis.png').astype(np.float)
		mask_gt = img_gt[:,:,0]
		mask_gt[mask_gt <= 250] = 0
		mask_gt[mask_gt > 250] = 1

		img_pred = cv2.imread(fold_pred+'/'+img_id[i]+ '_proj_dis.png').astype(np.float)
		mask_pred = img_pred[:,:,0]
		mask_pred[mask_pred <= 250] = 0
		mask_pred[mask_pred > 250] = 1		

		f_avg += db_eval_boundary(mask_pred, mask_gt)
	print(f_avg/len(img_id))

if __name__ == "__main__":
	F_Depth()