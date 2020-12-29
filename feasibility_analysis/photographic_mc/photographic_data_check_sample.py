import sys
import csv
from pathlib import Path
import random
import pickle

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *

pbanner()
psystem('Photographic track extractor')
pmode('Testing Feasibility')
pinfo('Input DType for testing: StrawDigiMC')

# load pickle
cwd = Path.cwd()
pickle_path = cwd.joinpath('photographic.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

sX = np.load(C.X_npy)
mY = np.load(C.Y_npy)
bboxNum = mY.shape[0]

for i in range(bboxNum):
    x = sX[i]
    y = mY[i]
    fig, (ax1, ax2) = plt.subplots(1,2)

    x = x.astype(np.uint8)
    ax1.imshow(255-255*x)

    mask_indices = np.where( (np.isnan(y)).all(axis=2) )
    major_indices = np.where( (y==(0,0,1)).all(axis=2) )
    bg_indices = np.where( (y==(0,1,0)).all(axis=2) )
    blank_indices = np.where( (y==(1,0,0)).all(axis=2) )
    y[mask_indices] = np.array((255,255,255), dtype=np.uint8)
    y[major_indices] = np.array((255,0,0), dtype=np.uint8)
    y[bg_indices] = np.array((0,0,255), dtype=np.uint8)
    y[blank_indices] = np.array((0,0,0), dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    ax2.imshow(y)


    #ax2.imshow(255-255*y)
    plt.show()
    plt.close()
