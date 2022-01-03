import sys
from pathlib import Path
import shutil
import timeit
import pickle

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database_new import *
from Configuration import wcnn_config
from Information import *


def preprocess(C):

    xfs = [f for f in C.X_train_dir.glob('*')]
    y1fs = [f for f in C.Y1_train_dir.glob('*')]
    y2fs = [f for f in C.Y2_train_dir.glob('*')]

    pos = 0.0
    neg = 0.0
    fNum = len(y1fs)
    for i, y1f in enumerate(y1fs):
        sys.stdout.write(t_info(f'Calculating weight: {(i+1)/float(fNum)*100}%', special='\r'))
        if i+1 == fNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        y1 = np.load(y1f)
        neg += np.sum(y1==0)
        pos += np.sum(y1==1)

    geo_avg = np.sqrt(pos*neg)
    weights = [pos/geo_avg, neg/geo_avg]
    pinfo(f"calculated weights: {weights}")
    C.set_weights(weights)

    return C

if __name__ == "__main__":
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('wcnn.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    start = timeit.default_timer()
    preprocess(C)
    total_time = timeit.default_timer()-start
    pinfo(f'Elapsed time: {total_time}(sec)')

    pickle.dump(C, open(pickle_path, 'wb'))
