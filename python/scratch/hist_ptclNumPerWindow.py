import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database_new import *
from Information import *
from HitGenerators import Event

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train_CeEndpoint-mix.db')]

# dist, db_files, hitNumCut=20):
gen = Event(db_files, 10)

def mean(ls):
    return sum(ls)/len(ls)

windowNum = 1
ptclNumPerWindows = []
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')

    t_avgs=[]
    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[0:-1]
        hits = [hit_all[hitIdx] for hitIdx in hitIdc]
        ts = [coord[3] for coord in hits]
        t_avgs.append(mean(ts))
    entries = np.histogram(t_avgs, 128)
    ptclNumPerWindow = entries[entries>0].tolist()
    ptclNumPerWindows.append(ptclNumPerWindow)

plt.hist(ptclNumPerWindows, 10)
plt.title('Histogram: Number of Particles Per Pre-set Window')
plt.xlabel('Number of Particles Per Pre-set Window')
plt.ylabel('Number of Windows')
plt.show()
