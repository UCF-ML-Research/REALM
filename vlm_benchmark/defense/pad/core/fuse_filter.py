import numpy as np
import cv2

from .heatmap_MI import img_heatmap_mi
from .heatmap_CD import img_heatmap_cd

ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_param = 80
thresh_param = 80 # percentile, from small to big

def fuse_heatmap(impath, ori_height, ori_width):
    h_mi = img_heatmap_mi(impath)
    h_cd, _qt = img_heatmap_cd(impath)
    h_cd = np.mean(h_cd, axis=0)

    h_mi = cv2.resize(h_mi, (ori_width, ori_height))
    h_cd = cv2.resize(h_cd, (ori_width, ori_height))

    h_mi_range = np.max(h_mi) - np.min(h_mi)
    h_mi_min = np.min(h_mi)
    h_mi = [int((h_mi[i][j]-h_mi_min)*255/h_mi_range) if h_mi_range > 0 else 0 for i in range(len(h_mi)) for j in range(len(h_mi[0]))]

    h_cd_range = np.max(h_cd) - np.min(h_cd)
    h_cd_min = np.min(h_cd)
    h_cd = [int((h_cd[i][j]-h_cd_min)*255/h_cd_range) if h_cd_range > 0 else 0 for i in range(len(h_cd)) for j in range(len(h_cd[0]))]

    h_fuse = [int(h_mi[i]*ratio_mi + h_cd[i]*(1-ratio_mi)) for i in range(len(h_mi))]

    h_fuse_grayImage = np.array(h_fuse, dtype=np.uint8).reshape(ori_height, ori_width)
    h_mi_grayImage = np.array(h_mi, dtype=np.uint8).reshape(ori_height, ori_width)
    h_cd_grayImage = np.array(h_cd, dtype=np.uint8).reshape(ori_height, ori_width)

    return h_mi_grayImage, h_cd_grayImage, h_fuse_grayImage

def heatmap_filter(heatmap, threshold, height, width):
    _thresh, h_t = cv2.threshold(heatmap, threshold, maxval=255, type=cv2.THRESH_TOZERO)

    base_kernel_size = int(min(height, width)/kernel_param)

    # MORPH_OPEN
    kernel = np.ones((base_kernel_size*2, base_kernel_size*2), np.uint8)
    h_t_o = cv2.morphologyEx(h_t, cv2.MORPH_OPEN, kernel, iterations=1)

    # MORPH_CLOSE
    kernel = np.ones((base_kernel_size, base_kernel_size), np.uint8)
    h_t_o_c = cv2.morphologyEx(h_t_o, cv2.MORPH_CLOSE, kernel, iterations=2)

    # MORPH_OPEN
    kernel = np.ones((base_kernel_size*3, base_kernel_size*3), np.uint8)
    h_t_o_c_o = cv2.morphologyEx(h_t_o_c, cv2.MORPH_OPEN, kernel, iterations=2)

    return h_t, h_t_o, h_t_o_c, h_t_o_c_o
