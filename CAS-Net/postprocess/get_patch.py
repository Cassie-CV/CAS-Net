#!/usr/bin/env python
# encoding: utf-8
'''
@author: caixia_dong
@license: (C) Copyright 2020-2023, Medical Artificial Intelligence, XJTU.
@contact: caixia_dong@xjtu.edu.cn
@software: MedAI
@file: get_patch.py
@time: 2022/7/22 14:55
@version:
@desc:
'''


def get_patch_new(img_shape, interval_0=128, interval_1=160, step=1):
    stepSize_0 = int(interval_0 / step)
    stepSize_1 = int(interval_1 / step)
    roi_list = []

    if img_shape[0] < interval_0 or img_shape[1] < interval_1 or img_shape[2] < interval_1:
        return []
    if img_shape[0] < interval_0:
        for i_row in range(0, img_shape[1] - (interval_1 - stepSize_1), stepSize_1):
            i_row_temp = i_row

            if i_row_temp + interval_1 >= img_shape[1]:
                i_row_temp = img_shape[1] - interval_1

            for i_col in range(0, img_shape[2] - (interval_1 - stepSize_1), stepSize_1):
                i_col_temp = i_col
                if i_col_temp + interval_1 >= img_shape[2]:
                    i_col_temp = img_shape[2] - interval_1
                roi_list.append([-1, i_row_temp, i_col_temp])
    else:
        for i_slice in range(0, img_shape[0] - (interval_0 - stepSize_0), stepSize_0):
            i_slice_temp = i_slice
            if i_slice + interval_0 >= img_shape[0]:
                i_slice_temp = img_shape[0] - interval_0
            # break
            # print (i_slice_temp)
            for i_row in range(0, img_shape[1] - (interval_1 - stepSize_1), stepSize_1):
                i_row_temp = i_row

                if i_row_temp + interval_1 >= img_shape[1]:
                    i_row_temp = img_shape[1] - interval_1

                for i_col in range(0, img_shape[2] - (interval_1 - stepSize_1), stepSize_1):
                    i_col_temp = i_col
                    if i_col_temp + interval_1 >= img_shape[2]:
                        i_col_temp = img_shape[2] - interval_1
                    roi_list.append([i_slice_temp, i_row_temp, i_col_temp])
    return roi_list