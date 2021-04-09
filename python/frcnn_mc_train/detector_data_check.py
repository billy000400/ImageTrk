import sys
from pathlib import Path
import shutil
import timeit
import pickle
import random
from copy import copy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Geometry import iou
from Information import *

def check_data(C, checkNum):

    img_files = [c for c in C.train_img_inputs_npy.iterdir()]
    imgNum = len(img_files)

    roi_files = [c for c in C.train_rois.iterdir()]

    for x in range(checkNum):
        idx = random.randint(0,imgNum-1)

        img = np.load(img_files[idx])
        roi = np.load(roi_files[idx])

        print('R:', np.count_nonzero(img[:,:,0]!=1))
        print('G:', np.count_nonzero(img[:,:,1]!=1))
        print('B:', np.count_nonzero(img[:,:,2]!=1))
        
        plt.imshow(img)
        plt.show()


    return

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Checking data for the Fast-RCNN detector')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    checkNum = 5

    C = check_data(C, 5)
