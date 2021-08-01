import sys
from pathlib import Path
from math import sqrt
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
    return PatchCollection(rects,label='anchor',linewidth=1, edgecolor='r', facecolor='none')

def get_anchor_points(res, ratio):
    x = [0.5+i*ratio for i in range(int(res/ratio))]
    y = deepcopy(x)
    x = [data for i in range(len(x)) for data in x]
    y = [data for data in y for i in range(len(y))]

    return x, y


if __name__=="__main__":

    ### prepare anchors
    anchor_scales = [0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
    anchor_ratios = [[1,1],\
                        [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]

    x=256.5
    y=256.5
    anchors = get_anchors(x,y,anchor_scales,anchor_ratios)
    anchor_pts = get_anchor_points(512,16)
    xs,ys = anchor_pts

    ### custom legend
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='anchor'),
                   Line2D([0], [0], marker='o', color='w', label='anchor center',
                          markerfacecolor='b', markersize=5)]

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.add_collection(anchors)
    ax.scatter(xs,ys, c='b', s=.5, label='anchor center')
    ax.set(xlim=[50,462], ylim=[52,462])
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()
