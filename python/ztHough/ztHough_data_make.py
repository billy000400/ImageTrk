# Make Training Set
# Author: Billy Haoyang Li

# General import
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Stochastic_reco


def plot_zt(generator, windowNum=100):
    for i in range(windowNum):
        hit_all, track_all = gen.generate(mode='eval')

        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[:-1]
            hits = [hit_all[hitIdx] for hitIdx in hitIdc]
            z = [vec[2] for vec in hits]
            t = [vec[3] for vec in hits]
            plt.scatter(z,t, label=str(trkIdx))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    pbanner()
    psystem('z-t Hough Transform')
    pmode('Generating z-t pictures')

    track_dir = Path("../../tracks")
    db_files = [track_dir.joinpath('val.db')]

    mean = 5.0
    std = 2.0
    dist = norm(loc=mean, scale=1/std)

    # dist, db_files, hitNumCut=20):
    gen = Stochastic_reco(dist, db_files)

    plot_zt(generator=gen, windowNum=100)
