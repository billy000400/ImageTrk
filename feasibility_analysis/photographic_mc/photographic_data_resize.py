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

pinfo("Loading raw numpy arrays")
X = np.load(C.X_file, allow_pickle=True)
Y = np.load(C.Y_file, allow_pickle=True)

# calculate scaled dimensions
res = C.resolution
scale_want = (res, int(res/4*3))
scale_alter = (int(res/4*3), res)

bboxNum = len(Y)
for i in range(bboxNum):
    xo=X[i]
    yo=Y[i]

    ratio = xo.shape[0]/xo.shape[1]
    im_in = Image.fromarray(xo)
    im_out = Image.fromarray(yo, mode='RGB')

    if ratio < 1:
        x = im_in.resize(scale_want)
        y = im_out.resize(scale_want)
    else:
        x = im_in.resize(scale_alter)
        y = im_out.resize(scale_alter)
        x = x.rotate(angle=90, expand=True)
        y = y.rotate(angle=90, expand=True)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(255-255*xo)
    ax2.imshow(255-255*np.array(x))
    ax3.imshow(255-255*yo)
    ax4.imshow(255-255*np.array(y))
    plt.show()
    plt.close()
