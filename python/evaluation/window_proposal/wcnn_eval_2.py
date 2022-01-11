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

        # predict windows
        map = hit2ztmap(hits_filtered, 256)
        windows_p, scores = wcnn.propose(map)
        windows_p = np.array(windows_p)
        print(windows_p.shape)
        print(windows_r.shape)

        plot_hits_and_windows(hits_filtered, trks_filtered, windows_r)
