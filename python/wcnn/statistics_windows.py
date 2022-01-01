import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event

def statistics_windows():
    track_dir = Path("../../tracks")
    db_files = [track_dir.joinpath('train_CeEndpoint-mix.db')]

    # target histogram
    delta_ts = []

    # dist, db_files, hitNumCut=20):
    windowNum = 500
    gen = Event(db_files, hitNumCut=10, eventNum=windowNum)

    for i in range(windowNum):
        hit_all, track_all = gen.generate(mode='eval')
        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[0:-1]
            hits = [hit_all[hitIdx] for hitIdx in hitIdc]
            ts = np.array([coord[3] for coord in hits], dtype=np.float)
            delta_t = ts.max()-ts.min()
            delta_ts.append(delta_t)

    entries, _ = np.histogram(delta_ts, 50)
    print(entries)
    plt.hist(delta_ts, 50)
    plt.show()
    return

if __name__ == "__main__":
    statistics_windows()
