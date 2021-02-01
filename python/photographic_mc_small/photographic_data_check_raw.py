import sys
import csv
from pathlib import Path
import random
import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Config import extractor_config as Config
from mu2e_output import *


pbanner()
psystem('Photographic track extractor')
pmode('Testing Feasibility: check raw data')
pinfo('Input DType for testing: StrawDigiMC')

# load pickle
cwd = Path.cwd()
pickle_path = cwd.joinpath('photographic.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

pinfo('Loading X Y arrays')
X = np.load(C.X_file, allow_pickle=True)
Y = np.load(C.Y_file, allow_pickle=True)

is_blank = np.array([1,0,0], dtype=np.int8)
is_bg = np.array([0,1,0], dtype=np.int8)
is_major = np.array([0,0,1], dtype=np.int8)

start = 1
count = 0

for x, y in zip(X,Y):

    if count < start:
        count += 1
        continue

    # calculate data limits
    ptcl_locations = np.where(x!=0)
    density = x[ptcl_locations]
    color = density*256/density.max()

    pos_x = deepcopy(ptcl_locations[0])
    pos_y = deepcopy(ptcl_locations[1])
    pos_x.sort()
    xlen = pos_x[-1]-pos_x[0]
    pos_y.sort()
    ylen = pos_y[-1]-pos_y[0]

    # calculating plot grid spec
    ul = 5
    ratio = ylen/xlen
    if ratio < 1:
        widths = [ul, ul]
        height = [ul*ratio]
    else:
        widths = [ul/ratio, ul/ratio]
        height = [ul]

    # plot the spec
    fig = plt.figure(figsize=(widths[0]*2,height[0]))
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths, height_ratios=height, wspace=0)

    # plot input photo
    ax1 = fig.add_subplot(spec[0,0])
    ax1.scatter(ptcl_locations[0], ptcl_locations[1], s=6, c=color)
    ax1.set(title='Input Density Map', xlabel='x pixel index', ylabel='y pixel index')
    ax1.set(xlim = [pos_x[0], pos_x[-1]], ylim=[pos_y[0], pos_y[-1]])

    # plot output truth
    major_locations = np.where(y[:,:,2]==1)
    if major_locations[0].size < 20:
        pdebug('Less than 13 major particles!')
    bg_locations = np.where(y[:,:,1]==1)
    ax2 = fig.add_subplot(spec[0,1], sharey=ax1)
    ax2.scatter(major_locations[0], major_locations[1], s=6, c='r', label='major')
    ax2.scatter(bg_locations[0], bg_locations[1], s=6, c='k', label='bg')
    ax2.legend()
    ax2.set(title='Output Truth Map', xlabel='x pixel index')
    ax2.set(xlim = [pos_x[0], pos_x[-1]], ylim=[pos_y[0], pos_y[-1]])

    # show the pic and close it after showing it
    plt.autoscale(enable=False, tight=True)
    plt.show()
    plt.close()
