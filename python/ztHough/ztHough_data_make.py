# Make Training Set
# Author: Billy Haoyang Li

# General import
import sys
from pathlib import Path

import cv2

import numpy as np
from scipy.stats import norm

from skimage.transform import hough_line, hough_line_peaks

from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event

def discretize(x, min, max, res):
    # Hard coded parameters
    step = (max-min)/res
    result = (x-min)//step
    return int(result)

def zt2map(zs, ts, res):
    map = np.zeros(shape=(res,res), dtype=float)
    zmin, zmax = -1600.0, 1600.0
    tmin, tmax = 0, 1800

    for z, t in zip(zs, ts):
        zIdx = discretize(z, zmin, zmax, res)
        tIdx = res-discretize(t, tmin, tmax, res)
        map[tIdx, zIdx] += 1

    max = map.max()
    map = map/max
    return map


def plot_zt(generator, windowNum=100):

    for i in range(windowNum):
        hit_all, track_all = gen.generate(mode='eval')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[:-1]
            hits = [hit_all[hitIdx] for hitIdx in hitIdc]
            z = [vec[2] for vec in hits]
            t = [vec[3] for vec in hits]

            ax1.scatter(z, t, label=str(trkIdx))
        ax1.legend()
        ax1.axis('scaled')
        ax1.axis('off')
        ax1.set(xlim=[-1600, 1600], ylim=[0,1800])

        zs = [coord[2] for idx, coord in hit_all.items()]
        ts = [coord[3] for idx, coord in hit_all.items()]

        map = zt2map(zs,ts,100)
        ax2.imshow(map)

        out, angles, d = hough_line(map, theta=np.linspace(0, 1, num=3))

        angle_step = 0.01
        d_step = 0.5 * np.diff(d).mean()
        bounds = (0,\
            1,\
            d[-1] + d_step, d[0] - d_step)

        ax3.imshow(out, cmap=plt.cm.bone, extent=bounds)

        ax4.imshow(map)
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())

        th = 0.43*out.max()

        for _, angle, dist in zip(*hough_line_peaks(out, angles, d, threshold=th)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            slope=np.tan(angle + np.pi/2)
            intercept = y0 - slope*x0
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals)
        plt.show()

        plt.close()
    return

if __name__ == "__main__":
    pbanner()
    psystem('z-t Hough Transform')
    pmode('Generating z-t pictures')

    track_dir = Path("../../tracks")
    db_files = [track_dir.joinpath('train_CeEndpoint-mix.db')]

    # dist, db_files, hitNumCut=20):
    gen = Event(db_files, 10)

    plot_zt(generator=gen, windowNum=100)
