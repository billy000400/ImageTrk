import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train_CeEndpoint-mix-fromCSV_1.db')]

# dist, db_files, hitNumCut=20):

# find the longest track
def findLongestTrk(tracks):
    idx = None
    lmax = 0
    for trkIdx, hits in tracks.items():
        l = len(hits)
        if l > lmax:
            idx=trkIdx
    return trkIdx

# find the smallest t among tracks
def find_tmin(tracks):
    ts = []
    for trkIdx, hits in tracks.items():
        for hit in hits:
            ts.append(hit[3])
    return min(ts)

windowNum = 100
trackNums = []

gen = Event(db_files, hitNumCut=13, totNum=windowNum)
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    trksLoaded = {}
    t_first_ptcl = 0
    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[0:-1]
        hits = [hit_all[hitIdx] for hitIdx in hitIdc]
        trksLoaded[trkIdx] = hits
        xs = np.array([coord[0] for coord in hits])
        ys = np.array([coord[1] for coord in hits])
        zs = np.array([coord[2] for coord in hits])
        ts = np.array([coord[3] for coord in hits])
        if t_first_ptcl==0:
            t_first_ptcl = min(ts)
            ax.scatter(xs, zs, ys, alpha=1, label=trkIdx)
            continue

        if max(ts)-t_first_ptcl < 90:
            ax.scatter(xs, zs, ys, alpha=1, label=trkIdx)
        else:
            if min(ts)-t_first_ptcl< 90:
                ax.scatter(xs[ts<(t_first_ptcl+90)],\
                            zs[ts<(t_first_ptcl+90)],\
                            ys[ts<(t_first_ptcl+90)], alpha=1, label=trkIdx)
            ax.legend()
            ax.set(xlim=[-810, 810], ylim=[-1600, 1600], zlim=[-810, 810])

            plt.show()
            plt.close()

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # find the longest track
            longestTrkIdx = findLongestTrk(trksLoaded)

            # delete the track
            del trksLoaded[longestTrkIdx]

            # update t_first_ptcl
            t_first_ptcl = find_tmin(trksLoaded)
