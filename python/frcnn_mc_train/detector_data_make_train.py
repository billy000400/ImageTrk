### Make Trianing set for the Fast-RCNN detector

import sys
from pathlib import Path
import shutil
import timeit
import pickle

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from TrackDB_Classes import *
from Config import frcnn_config as Config
from Abstract import binning_objects
from mu2e_output import *

def make_data(C):

    # read region proposals from the prediction file
    df = pd.read_csv(C.bbox_prediction_file, index_col=0)
    roiNum = len(df.index)

    # register memory for input and output data
    rowNum = C.input_shape[0]/C.base_net.ratio
    colNum = C.input_shape[1]/C.base_net_ratio
    inputs = C.inputs_npy
    outputs_classifier = np.zeros(shape=(roiNum, 1))
    outputs_regressor = np.zeros(shape=(roiNum, 4))

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Making data for the Fast-RCNN detector')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    make_data(C)
