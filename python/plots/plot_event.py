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


windowNum = 100
trackNums = []

gen = Event(db_files, hitNumCut=20, totNum=windowNum)
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[0:-1]
        hits = [hit_all[hitIdx] for hitIdx in hitIdc]
        xs = [coord[0] for coord in hits]
        ys = [coord[1] for coord in hits]
        zs = [coord[2] for coord in hits]
        ax.scatter(xs, zs, ys, alpha=1, label=trkIdx)

    ax.legend()
    ax.set(xlim=[-810, 810], ylim=[-1600, 1600], zlim=[-810, 810])

    plt.show()
    plt.close()
