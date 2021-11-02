import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Stochastic

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train.db')]

mean = 5.0
std = 2.0
dist = norm(loc=mean, scale=1/std)

# dist, db_files, hitNumCut=20):
gen = Stochastic(dist, db_files)

for i in range(20):
    hit_all, track_all = gen.generate(mode='eval')

    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[:-1]
        hits = [hit_all[hitIdx] for hitIdx in hitIdc]
        z = [vec[2] for vec in hits]
        t = [vec[3] for vec in hits]
        plt.scatter(z,t, label=str(trkIdx))
    plt.legend()
    plt.show()
