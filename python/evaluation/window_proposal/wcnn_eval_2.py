# @Author: Billy Li <billyli>
# @Date:   01-11-2022
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 01-11-2022



### This script is to visually compare the ground truth and the prediction
### made by WCNN
import sys
from pathlib import Path
import shutil
import timeit
import pickle
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event
from Abstract import *
from Architectures import WCNN

def take_score(w_p):
    return w_p[2]

def union1D(intv_a, intv_b, intersection):
    w_a = intv_a[1]-intv_a[0]
    w_b = intv_b[1]-intv_b[0]
    return w_a+w_b-intersection

def intersection1D(intv_a, intv_b):
    # rec_a(b) should be (xmin, xmax, ymin, ymax)
    w = min([intv_a[1], intv_b[1]]) - max([intv_a[0], intv_b[0]])
    return max([w, 0])

def iou1D(intv_a, intv_b):
    overlap = intersection1D(intv_a, intv_b)
    sum = union1D(intv_a, intv_b, overlap)
    return overlap/sum


def ap_1D(ws_t, ws_p, threshold=0.9):
    ws_p_ordered = sorted(ws_p, key=take_score)
    all_positive = len(ws_t)
    TP, FP = 0, 0
    precisions, recalls = [1.0], [0.0]
    for w_p in ws_p_ordered:
        iou_max = 0.0
        argmax = None
        for i, w_t in enumerate(ws_t):
            iou = iou1D(w_t, w_p)
            if iou > iou_max:
                iou_max = iou
                argmax = i
        if iou_max < threshold:
            FP += 1
        else:
            TP += 1
            precisions.append(float(TP)/(TP+FP))
            recalls.append(TP/all_positive)
            del ws_t[argmax]

    precisions.append(0.0)
    recalls.append(1.0)

    ap = 0.0
    for i in range(len(recalls)-1):
        dx = recalls[i+1] - recalls[i]
        y = precisions[i+1]
        ap += dx*y

    return ap

def plot_hits_and_windows(hit_all, track_all, windows):

    zss = []
    tss = []
    for trkIdx, hitIdc in track_all.items():
        hitsInTrack = [hit_all[idx] for idx in hitIdc]
        zs = [hit[2] for hit in hitsInTrack]
        ts = [hit[3] for hit in hitsInTrack]
        zss.append(zs)
        tss.append(ts)

    for i,bds in enumerate(windows):
        for zs, ts in zip(zss,tss):
            plt.scatter(zs, ts, s=3)
        min, max = bds
        plt.axhline(min*(1840)-40, lw=0.5)
        plt.axhline(max*(1840)-40, lw=0.5)
        plt.ylim(min*(1840)-40-5, max*(1840)-40+5)
        plt.show()
        plt.close()

    return


if __name__ == "__main__":
    track_dir = Path.cwd().parent.parent.parent.joinpath('tracks')
    dp_name = 'train_CeEndpoint-mix'
    dp_file = track_dir.joinpath(dp_name+'.db')
    gen = Event([dp_file], 10, eventNum=50)

    res = 256
    scales = [float(1/256)]
    weight_path = "wcnn_02_tanh.h5"
    wcnn = WCNN(resolution=256, anchor_scales=scales, weight_path=weight_path)
    testNum = 10

    for i in range(testNum):

        # generate an event
        hit_all, track_all = gen.generate(mode='eval')
        # filter out long staying ptcls
        trks_filtered = {}
        hits_filtered = {}
        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[:-1]
            hitsPerTrack = [hit_all[idx] for idx in hitIdc]
            tsPerTrack = [hit[3] for hit in hitsPerTrack]
            tmin = min(tsPerTrack)
            tmax = max(tsPerTrack)
            delta_t= tmax-tmin
            if delta_t > 1000:
                continue
            trks_filtered[trkIdx] = hitIdc
            for idx in hitIdc:
                hits_filtered[idx]=hit_all[idx]

        # obtain window truth
        windows_r = trk2windows(hits_filtered, trks_filtered)
        windows_r = np.array(windows_r)
        gts = windows_r.tolist()

        # predict windows
        map = hit2ztmap(hits_filtered, 256)
        windows_p, scores = wcnn.propose(map)
        windows_p = np.array(windows_p)
        scores = np.array(scores)

        preds = []
        for window_p, score in zip(windows_p, scores):
            preds.append([window_p[0], window_p[1], score])

        ap = ap_1D(gts, preds)
        print(ap)

    print("over")
        # print(windows_p.shape)
        # print(windows_r.shape)

        # plot_hits_and_windows(hits_filtered, trks_filtered, windows_r)
