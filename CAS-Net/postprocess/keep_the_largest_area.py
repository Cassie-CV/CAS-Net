#!/usr/bin/env python
# encoding: utf-8
'''
@author: caixia_dong
@license: (C) Copyright 2020-2023, Medical Artificial Intelligence, XJTU.
@contact: caixia_dong@xjtu.edu.cn
@software: MedAI
@file: keep_the_largest_area.py
@time: 2022/7/22 14:49
@version:
@desc:
'''

from __future__ import division
import os
import numpy as np
import pydicom
from skimage import measure
import copy
import numpy as np
import cv2
import re
import os
import copy
from skimage import measure
from scipy import ndimage

def get_aorta_branch(mask):
    mask[mask>0]=1
    mask_filter_erode = get_mask_filter_erode(mask)
    mask_filter_erode = mask_track(mask_filter_erode)

    mask_branch = mask - mask_filter_erode
    mask_branch = part_filter(mask_branch)
    return mask_filter_erode,mask_branch

###贴标签，保留最大面积
def backpreprcess(ret_box):
    #Label后处理
    box = []
    [heart_res, num] = measure.label(ret_box, return_num = True)
    region = measure.regionprops(heart_res)
    for i in range(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    heart_res[heart_res != label_num] = 0
    heart_res[heart_res == label_num] = 1
    heart_res = np.array(heart_res, dtype = 'uint8')
    return heart_res

def label_filter(mask):
    mask_cp = copy.deepcopy(mask)
    try:
        image, contours, hierarchy = cv2.findContours(mask_cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(mask_cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    i_max = 0
    for i in range(len(contours)):
        contour = contours[i]
        retval = cv2.contourArea(contour)
        if (retval > max_area):
            max_area = retval
            i_max = i
    c_min = []
    for i in range(len(contours)):
        if i != i_max:
            c_min.append(contours[i])

    cv2.drawContours(mask_cp, c_min, -1, (0, 0, 0), cv2.FILLED)
    return mask_cp

def get_mask_filter(mask):

    mask_cp = copy.deepcopy(mask)
    slice_num = mask_cp.shape[0]
    mask_filter = np.zeros((slice_num, 512, 512), dtype=np.uint8)
    for i in range(slice_num):
        mask_slice = mask_cp[i, :, :]
        mask_temp = label_filter(mask_slice)
        mask_filter[i, :, :] = mask_temp

    return  mask_filter

def get_mask_filter_erode(mask, erode_size = 15):

    mask_cp = copy.deepcopy(mask)
    mask_filter = np.zeros(mask_cp.shape, dtype=np.uint8)
    for i in range(mask_cp.shape[0]):
        mask_slice = mask_cp[i, :, :]

        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        erosion = cv2.erode(mask_slice, erode_kernel)
        mask_temp = label_filter(erosion)
        dilate_kernel = np.ones((erode_size, erode_size), np.uint8)
        dilation = cv2.dilate(mask_temp, dilate_kernel)

        mask_filter[i, :, :] = dilation
    mask_filter_out = mask_filter & mask
    return  mask_filter_out

def mask_track(mask):
    mask_cp = copy.deepcopy(mask)
    mask_out = np.zeros(mask_cp.shape, dtype=np.uint8)

    mask_up = np.zeros((512, 512), dtype=np.uint8)
    is_up_find = 0
    for i in range(mask_cp.shape[0]):
        mask_slice = mask_cp[0, :, :]
        if np.max(mask_slice) > 0 and is_up_find == 0:
            mask_up = mask_slice
            is_up_find = 1
        if is_up_find:
            mask_temp = mask_slice & mask_up
            if np.max(mask_temp) > 0:
                mask_up = mask_cp[i, :, :]
            else:
                break
        mask_out[i, :, :] = mask_cp[i, :, :]

    return mask_out

def part_filter(mask):
    mask_cp = copy.deepcopy(mask)
    [heart_res, num] = measure.label(mask_cp, return_num=True)

    if num <= 2:
        return mask_cp

    region = measure.regionprops(heart_res)
    box = []
    for i in range(num):
        box.append(region[i].area)

    # print (box)
    label_num = box.index(max(box)) + 1
    # print (label_num)
    mask_cp[heart_res != label_num] = 0
    mask_cp[heart_res == label_num] = 1

    box[box.index(max(box))] = 0
    # print(box)
    label_num = box.index(max(box)) + 1
    # print (label_num)
    mask_cp[heart_res == label_num] = 1

    mask_cp = np.array(mask_cp, dtype='uint8')

    return mask_cp
