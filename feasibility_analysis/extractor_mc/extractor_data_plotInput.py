# import starts
import sys
from pathlib import Path
import csv
import random
import pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *
### import ends

# load pickle
cwd = Path.cwd()
pickle_path = cwd.joinpath('extractor.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

### load inputs
pinfo('Loading processed arrays')
X = np.load(C.X_npy)
Y = np.load(C.Y_npy)

print(X.shape)
print(Y.shape)

for i in range(X.shape[0]):
    positions = X[i]
    truth_table = Y[i]

    target_x, target_y, target_z = [ [] for i in range(3) ]
    other_x, other_y, other_z = [ [] for i in range(3) ]
    for j, truth in enumerate(truth_table):
        if truth == True:
            target_x.append(X[i][j][0])
            target_y.append(X[i][j][1])
            target_z.append(X[i][j][2])
        else:
            other_x.append(X[i][j][0])
            other_y.append(X[i][j][1])
            other_z.append(X[i][j][2])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(target_z, target_x, target_y, label='target')
    ax.scatter(other_z, other_x, other_y, label='background')
    ax.legend()
    plt.show()
    plt.close()
