#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30
Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''
import os
import numpy as np
import cv2

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl
    count_class = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) < 1000) or (np.sum(curr_gt_mask) <1000):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
        count_class += 1
 
    mean_IU_ = np.sum(IU) / count_class
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) < 1000) or (np.sum(curr_gt_mask) <1000):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c        

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == "__main__":
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
    
    pix_accu_cvpr_avg = 0.0
    pix_accu_p2p_avg = 0.0
    pix_accu_our_avg = 0.0
    pix_accu_abl1_avg = 0.0
    pix_accu_abl2_avg = 0.0
    pix_accu_abl3_avg = 0.0
    mean_IU_cvpr = 0.0
    mean_IU_p2p = 0.0
    mean_IU_our = 0.0
    mean_IU_abl1 = 0.0
    mean_IU_abl2 = 0.0
    mean_IU_abl3 = 0.0
    frequency_weighted_IU_cvpr = 0.0
    frequency_weighted_IU_p2p = 0.0
    frequency_weighted_IU_our = 0.0
    frequency_weighted_IU_abl1 = 0.0
    frequency_weighted_IU_abl2 = 0.0
    frequency_weighted_IU_abl3 = 0.0
    for i in range(0,len(img_id)):
        print(i)
        img_gt = cv2.imread(fold_gt+'/'+img_id[i]+ '_street_rgb_00.png').astype(np.int)
        mask_gt = img_gt[:,:,0] + img_gt[:,:,1]+ img_gt[:,:,2]

        img_cvpr = cv2.imread(fold_cvpr+'/'+img_id[i]+ '_cvpr2018_00.png').astype(np.int)
        mask_cvpr = img_cvpr[:,:,0] + img_cvpr[:,:,1]+ img_cvpr[:,:,2]

        img_p2p = cv2.imread(fold_p2p+'/'+img_id[i]+ '_cvpr2018_00.png').astype(np.int)
        mask_p2p = img_p2p[:,:,0]+ img_p2p[:,:,1] + img_p2p[:,:,2]

        img_our = cv2.imread(fold_our+'/'+img_id[i]+ '.png').astype(np.int)
        mask_our = img_our[:,:,0]+ img_our[:,:,1] + img_our[:,:,2]

        img_abl1 = cv2.imread(fold_abl_1+'/'+img_id[i]+ '.png').astype(np.int)
        mask_abl1 = img_abl1[:,:,0] + img_abl1[:,:,1]+ img_abl1[:,:,2]

        img_abl2 = cv2.imread(fold_abl_2+'/'+img_id[i]+ '.png').astype(np.int)
        mask_abl2 = img_abl2[:,:,0] + img_abl2[:,:,1]+ img_abl2[:,:,2]

        img_abl3 = cv2.imread(fold_abl_3+'/'+img_id[i]+ '.png').astype(np.int)
        mask_abl3 = img_abl3[:,:,0] + img_abl3[:,:,1]+ img_abl3[:,:,2]
        
        # pix accuracy
        pix_accu_cvpr_avg += pixel_accuracy(mask_cvpr, mask_gt)
        pix_accu_p2p_avg += pixel_accuracy(mask_p2p, mask_gt)
        pix_accu_our_avg += pixel_accuracy(mask_our, mask_gt)
        pix_accu_abl1_avg += pixel_accuracy(mask_abl1, mask_gt)
        pix_accu_abl2_avg += pixel_accuracy(mask_abl2, mask_gt)
        pix_accu_abl3_avg += pixel_accuracy(mask_abl3, mask_gt)
        
        # mean IU
        mean_IU_cvpr += mean_IU(mask_cvpr, mask_gt)
        mean_IU_p2p += mean_IU(mask_p2p, mask_gt)
        mean_IU_our += mean_IU(mask_our, mask_gt)
        mean_IU_abl1 += mean_IU(mask_abl1, mask_gt)
        mean_IU_abl2 += mean_IU(mask_abl2, mask_gt)
        mean_IU_abl3 += mean_IU(mask_abl3, mask_gt)

        # frequency_weighted_IU
        frequency_weighted_IU_cvpr += frequency_weighted_IU(mask_cvpr, mask_gt)
        frequency_weighted_IU_p2p += frequency_weighted_IU(mask_p2p, mask_gt)
        frequency_weighted_IU_our += frequency_weighted_IU(mask_our, mask_gt)
        frequency_weighted_IU_abl1 += frequency_weighted_IU(mask_abl1, mask_gt)
        frequency_weighted_IU_abl2 += frequency_weighted_IU(mask_abl2, mask_gt)
        frequency_weighted_IU_abl3 += frequency_weighted_IU(mask_abl3, mask_gt)
        
    print(pix_accu_cvpr_avg/len(img_id))
    print(pix_accu_p2p_avg/len(img_id))
    print(pix_accu_our_avg/len(img_id))
    print(pix_accu_abl1_avg/len(img_id))
    print(pix_accu_abl2_avg/len(img_id))
    print(pix_accu_abl3_avg/len(img_id))

    print(mean_IU_cvpr/len(img_id))
    print(mean_IU_p2p/len(img_id))
    print(mean_IU_our/len(img_id))
    print(mean_IU_abl1/len(img_id))
    print(mean_IU_abl2/len(img_id))
    print(mean_IU_abl3/len(img_id))

    print(frequency_weighted_IU_cvpr/len(img_id))
    print(frequency_weighted_IU_p2p/len(img_id))    
    print(frequency_weighted_IU_our/len(img_id)) 
    print(frequency_weighted_IU_abl1/len(img_id)) 
    print(frequency_weighted_IU_abl2/len(img_id)) 
    print(frequency_weighted_IU_abl3/len(img_id))    


