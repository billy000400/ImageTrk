import sys
import csv
from pathlib import Path
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *


pbanner()
psystem('Photographic track extractor')
pmode('Testing Feasibility: check raw data')
pinfo('Input DType for testing: StrawDigiMC')

# load pickle
cwd = Path.cwd()
pickle_path = cwd.joinpath('photographic.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

X = np.load(C.X_file)
Y = np.load(C.Y_file)
Y = np.asarray(Y, dtype=np.int8)

for x, y in zip(X,Y):
    ptcl_locations = np.where(x!=0)
    density = x[ptcl_locations]
    color = density*256/density.max()

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5) )
    ax1.scatter(ptcl_locations[0], ptcl_locations[1], s=1, c=color)
    ax2.imshow(y, interpolation='none')
    plt.show()
