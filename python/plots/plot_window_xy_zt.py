import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_MC as Event

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train_CeEndpoint-mix.db')]

# dist, db_files, hitNumCut=20):


def cluster_hits_by_time(hit_all):
    clusters = []
    t_prev = -1
    current_cluster = []
    hits = [hit for hitIdx, hit in hit_all.items()]
    hits.sort(key=lambda hit: hit[3])
    for hit in hits:
        t_curr = hit[3]
        if t_prev==-1:
            t_prev=t_curr
            current_cluster.append(hit)
            continue
        delta_t = t_curr - t_prev
        t_prev = t_curr
        if delta_t > 30:
            clusters.append(current_cluster)
            current_cluster=[hit]
        else:
            current_cluster.append(hit)
    clusters.append(current_cluster)

    return clusters

windowNum = 100
gen = Event(db_files, digiNumCut=13, totNum=windowNum)
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')

    while len(hit_all)!=0:
        # find hits in first 100 ns
        ts = [hit[3] for hitId, hit in hit_all.items()]
        tmin = min(ts)
        hitIds_selected = []
        for hitId, hit in hit_all.items():
            t = hit[3]
            if t-tmin > 100:
                continue
            else:
                hitIds_selected.append(hitId)

        # find which track they belong
        tracks_selected = {}
        for trkIdx, hitIdsPdgId in track_all.items():
            hitIds = hitIdsPdgId[0:-1]
            hitsInWindow = []
            for hitId in hitIds:
                if hitId in hitIds_selected:
                    hitsInWindow.append(hit_all[hitId])
            if len(hitsInWindow)!=0:
                tracks_selected[trkIdx] = hitsInWindow

        # plot them
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

        for trkIdx, hitsInWindow in tracks_selected.items():
            xs = [hit[0] for hit in hitsInWindow]
            ys = [hit[1] for hit in hitsInWindow]
            zs = [hit[2] for hit in hitsInWindow]
            ax1.scatter(xs, ys, alpha=1, label=trkIdx, s=2)

        ax1.set(xlim=[-810, 810], ylim=[-810, 810], xlabel='x', ylabel='y')
        ax1.legend()

        plt.show()
        plt.close()

        # find which track has most hits in 100 ns
        hitNumMax = 0
        longestTrkIdx = None
        for trkIdx, hitsInWindow in tracks_selected.items():
            if len(hitsInWindow)>hitNumMax:
                longestTrkIdx = trkIdx
                hitNumMax = len(hitsInWindow)

        # delete that track's hits
        hitIds_deleting = track_all[longestTrkIdx][0:-1]
        for hitId in hitIds_deleting:
            del hit_all[hitId]
