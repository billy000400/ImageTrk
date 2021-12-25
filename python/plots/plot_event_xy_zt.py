import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train_CeEndpoint-mix.db')]

# dist, db_files, hitNumCut=20):
gen = Event(db_files, 10)

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

    return clusters


windowNum = 100
trackNums = []
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[0:-1]
        hits = [hit_all[hitIdx] for hitIdx in hitIdc]
        xs = [coord[0] for coord in hits]
        ys = [coord[1] for coord in hits]
        zs = [coord[2] for coord in hits]
        ts = [coord[3] for coord in hits]
        ax3.scatter(xs, ys, alpha=1, label=trkIdx)
        ax1.scatter(zs, ts, alpha=1, label=trkIdx)

    clusters = cluster_hits_by_time(hit_all)
    for clusterIdx, cluster in enumerate(clusters):
        xs = [coord[0] for coord in cluster]
        ys = [coord[1] for coord in cluster]
        zs = [coord[2] for coord in cluster]
        ts = [coord[3] for coord in cluster]
        ax2.scatter(zs, ts, alpha=1, label=clusterIdx)

    ax3.legend()
    ax3.set(xlim=[-810, 810], ylim=[-810, 810])

    ax1.legend()

    plt.show()
    plt.close()
