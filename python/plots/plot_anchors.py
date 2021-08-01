import sys
from pathlib import Path
from math import sqrt
from copy import deepcopy

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from HitGenerators import Stochastic

def get_pixels(n):
    segs_h = [((0,i),(n,i)) for i in range(n)]
    segs_v = [((i,0),(i,n)) for i in range(n)]
    segs = segs_h+segs_v
    line_segments = LineCollection(segs, linewidth=.01, colors='grey',\
                                label='pixel boundary')
    return line_segments

def get_anchors(x,y,anchor_scales,anchor_ratios):
    res = 512
    rects = []
    for aks in anchor_scales:
        for akr in anchor_ratios:
            width = akr[0]*aks*res
            height = akr[1]*aks*res
            rec_x = x-width*0.5
            rec_y = y-height*0.5
            rec_xy = (rec_x,rec_y)
            rect = Rectangle(rec_xy,width,height)
            rects.append(rect)
    return rects

def get_anchor_points(res, ratio):
    x = [0.5+(i+1)*ratio for i in range(int(res/ratio)-1)]
    y = deepcopy(x)
    x = [data for i in range(len(x)) for data in x]
    y = [data for data in y for i in range(len(y))]

    return x, y


if __name__=="__main__":

    ### generate a tracking window's signal
    track_dir = Path('/home/Billy/Mu2e/analysis/DLTracking/tracks')
    dp_list =  ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art"]

    db_list = [ track_dir.joinpath(dp+'.db') for dp in dp_list]

    mean = 5.0
    std = 2.0
    dist = norm(loc=mean, scale=1/std)

    gen = Stochastic(dist=dist, db_files=db_list, hitNumCut=20)
    hits = gen.generate(mode='production')
    def normalize(x):
        return (x+810)/1620*512
    hits_x = [normalize(hits[id][0]) for id in hits]
    hits_y = [normalize(hits[id][1]) for id in hits]

    ### prepare anchors
    anchor_scales = [0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
    anchor_ratios = [[1,1],\
                        [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]

    anchor_pts = get_anchor_points(512,16)
    xs,ys = anchor_pts

    rects = []
    for x,y in zip(xs,ys):
        rects_tmp = get_anchors(x,y,anchor_scales,anchor_ratios)
        rects = rects+rects_tmp

    anchors = PatchCollection(rects,label='anchor',linewidth=.01, edgecolor='r', facecolor='none')

    image_edge = Rectangle((0,0),512,512, facecolor='none', edgecolor='k')

    ### custom legend
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='anchor'),
                        Line2D([0], [0], color='k', lw=2, label='image edge'),
                   Line2D([0], [0], marker='o', color='w', label='anchor center',
                          markerfacecolor='b', markersize=5),\
                          Line2D([0], [0], marker='o', color='w', label='hit',
                                 markerfacecolor='black', markersize=5)]

    fig, ax = plt.subplots(1,1,figsize=(8,8))

    # img = mpimg.imread('00002.png')
    # ax.imshow(img, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
    ax.add_collection(anchors)
    ax.add_patch(image_edge)
    ax.scatter(hits_x, hits_y, c='k', s=1)
    ax.scatter(xs,ys, c='b', s=.5, label='anchor center')

    ax.set(xlim=[-150,662], ylim=[-150,662], xlabel='X pixel', ylabel='Y pixel')
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()
