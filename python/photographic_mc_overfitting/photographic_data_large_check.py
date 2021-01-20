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
from Config import extractor_config as Config
from mu2e_output import *

pbanner()
psystem('Photographic track extractor')
pmode('Testing Feasibility')
pinfo('Input DType for testing: StrawDigiMC')

# load pickle
cwd = Path.cwd()
pickle_path = cwd.joinpath('photographic.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

X_dir = C.X_dir
Y_dir = C.Y_dir
xs = [child for child in X_dir.iterdir()]
ys = [child for child in Y_dir.iterdir()]

for i in range(len(xs)):
    x = xs[i]
    y = ys[i]

    x = np.load(x).astype(np.uint8)
    y = np.load(y).astype(np.uint8)

    x_max = int(x.max())
    ratio = 255/x_max
    for n in range(x_max+1):
        x[x==n] = np.uint8(255-n*ratio)

    y[(y==[1,0,0]).all(axis=2)] = np.array([255,255,255], dtype=np.uint8)
    y[(y==[0,1,0]).all(axis=2)] = np.array([0,0,255], dtype=np.uint8)
    y[(y==[0,0,1]).all(axis=2)] = np.array([255,0,0], dtype=np.uint8)

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(x, cmap='gray')
    ax2.imshow(y)


    #ax2.imshow(255-255*y)
    plt.show()
    plt.close()
