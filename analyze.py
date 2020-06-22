#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import shapely.geometry as sp
import matplotlib.pyplot as plt


def distance(a, b):
    return np.math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def xywh2xyxy(rect):
    return np.array([
        rect[0], rect[1], rect[2] + rect[0] - 1,
                          rect[3] + rect[1] - 1
    ])


def cxywh2xywh(box):
    return np.array([
        box[0] - (box[2] - 1) / 2, box[1] - (box[3] - 1) / 2, box[2], box[3]
    ])


def xyxy_2_4xy(box):
    return np.array([
        box[0], box[1], box[0], box[3], box[2], box[3], box[2], box[1]
    ])


def get_speed(f_gt):
    len_file = len(f_gt)
    list_coord = []
    for i in range(len_file):
        data = f_gt[i]
        coord = [(data[0] + data[4]) / 2, (data[1] + data[5]) / 2]
        list_coord.append(coord)
    list_speed = []
    for i in range(1, len_file):
        list_speed.append(distance(list_coord[i], list_coord[i - 1]))
    return list_speed


def cal_overlap_ratio(gt, res) -> int:
    if len(res) != 4:
        return -1
    res_xyxy = xywh2xyxy(res)
    res_xyxyxyxy = xyxy_2_4xy(res_xyxy)
    gt_rect = sp.Polygon(np.array(gt).reshape(4, 2))
    gt_area = gt_rect.area
    res_rect = sp.Polygon(res_xyxyxyxy.reshape(4, 2))
    intersect_area = gt_rect.intersection(res_rect).area

    return intersect_area / gt_area


def cal_area(gt):
    gt_rect = sp.Polygon(np.array(gt).reshape(4, 2))
    gt_area = gt_rect.area
    return gt_area

def main():
    with open('list.txt') as f:
        vot_list = f.read().splitlines()
    f.close()
    gt_list = []
    res_list = []
    speed_list = []
    ratio_list = []
    area_list = []
    for i in vot_list:
        cur_gt_list = []
        cur_res_list = []
        with open('vot2019/VOT2019/%s/groundtruth.txt' % i) as f:
            for line in f.readlines():
                inter_line = list(map(float, line.strip().split(',')))
                cur_gt_list.append(inter_line)
        f.close()
        gt_list.append(cur_gt_list)
        with open('siamfcpp_alexnet/baseline/%s/%s_001.txt' % (i, i)) as f:
            for line in f.readlines():
                inter_line = list(map(float, line.strip().split(',')))
                cur_res_list.append(inter_line)
        f.close()
        res_list.append(cur_res_list)
    for i in range(59):
        speed_list.append(get_speed(gt_list[i]))
        cur_ratio_list = []
        cur_area_list = []
        for j in range(1, len(gt_list[i])):
            cur_ratio_list.append(cal_overlap_ratio(gt_list[i][j], res_list[i][j]))
            cur_area_list.append(cal_area(gt_list[i][j]))
        ratio_list.append(cur_ratio_list)
        area_list.append(cur_area_list)

    index = 0
    for i in vot_list:
        plt.cla()
        sp_arr = np.array(area_list[index])
        ra_arr = np.array(ratio_list[index])
        approx = np.polyfit(sp_arr, ra_arr, 1)
        y_eval = np.polyval(approx, sp_arr)
        plt.scatter(area_list[index], ratio_list[index], label= 'original')
        plt.plot(area_list[index], y_eval, label = 'approximate')
        plt.xlabel('Area of ground truth')
        plt.ylabel('Overlap ratio')
        plt.savefig('alexnet_result_gtarea/%s.jpg'%i)
        index = index+1

    # index = 0
    # for i in vot_list:
    #     cur_cm = []
    #     cur_ill = []
    #     cur_mot = []
    #     cur_occ = []
    #     cur_size = []
    #
    #     try:
    #         with open('vot2019/VOT2019/%s/occlusion.tag' % i) as f:
    #             for line in f.readlines():
    #                 temp = int(line.strip('\n'))
    #                 cur_occ.append(temp)
    #         f.close()
    #         cur_occ.pop(0)
    #         plt.cla()
    #         plt.scatter(cur_occ, ratio_list[index])
    #         plt.xlabel('Occlution')
    #         plt.ylabel('Overlap ratio')
    #         plt.savefig('alexnet_result_occlusion/%s.jpg' % i)
    #     except:
    #         pass
    #
    #     with open('vot2019/VOT2019/%s/illum_change.tag' % i) as f:
    #         for line in f.readlines():
    #             temp = int(line.strip('\n'))
    #             cur_ill.append(temp)
    #     f.close()
    #     cur_ill.pop(0)
    #     plt.cla()
    #     plt.scatter(cur_ill, ratio_list[index])
    #     plt.xlabel('illumination change')
    #     plt.ylabel('Overlap ratio')
    #     plt.savefig('alexnet_result_illumchange/%s.jpg' % i)
    #
    #     try:
    #         with open('vot2019/VOT2019/%s/motion_change.tag' % i) as f:
    #             for line in f.readlines():
    #                 temp = int(line.strip('\n'))
    #                 cur_mot.append(temp)
    #         f.close()
    #         cur_mot.pop(0)
    #         plt.cla()
    #         plt.scatter(cur_mot, ratio_list[index])
    #         plt.xlabel('motion change')
    #         plt.ylabel('Overlap ratio')
    #         plt.savefig('alexnet_result_motchange/%s.jpg' % i)
    #     except:
    #         pass
    #
    #     try:
    #         with open('vot2019/VOT2019/%s/camera_motion.tag' % i) as f:
    #             for line in f.readlines():
    #                 temp = int(line.strip('\n'))
    #                 cur_cm.append(temp)
    #         f.close()
    #         cur_cm.pop(0)
    #         plt.cla()
    #         plt.scatter(cur_cm, ratio_list[index])
    #         plt.xlabel('camera motion')
    #         plt.ylabel('Overlap ratio')
    #         plt.savefig('alexnet_result_cammotion/%s.jpg' % i)
    #     except:
    #         pass
    #
    #     try:
    #         with open('vot2019/VOT2019/%s/size_change.tag' % i) as f:
    #             for line in f.readlines():
    #                 temp = int(line.strip('\n'))
    #                 cur_size.append(temp)
    #         f.close()
    #         cur_size.pop(0)
    #         plt.cla()
    #         plt.scatter(cur_size, ratio_list[index])
    #         plt.xlabel('size_change')
    #         plt.ylabel('Overlap ratio')
    #         plt.savefig('alexnet_result_sizechange/%s.jpg' % i)
    #     except:
    #         pass
    #
    #     index += 1


if __name__ == "__main__":
    main()
