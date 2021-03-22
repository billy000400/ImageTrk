# A Faster R-CNN approach to Mu2e Tracking
# The detector (Fast R-CNN) part for alternative (i) method in the original paper
# see https://arxiv.org/abs/1506.01497
# Author: Billy Haoyang Li
# Email: li000400@umn.edu

### imports starts
import sys
from pathlib import Path
import pickle
import timeit

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *
from Config import frcnn_config as Config
from Loss import *
from Metric import *
### imports ends

def detector_train(C):
    pstage("Start Training")

    # prepare the tensorflow.data.DataSet object
    inputs = np.load(C.inputs_npy)
    classifiers = np.load(C.detector_train_Y_classifier)
    regressors = np.load(C.detector_train_Y_regressor)

    # outputs
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    weights_dir = cwd.parent.parent.joinpath('weights')

    model_weights = weights_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    pinfo('I/O Path is configured')

    # build the model
    img_input = Input(shape=C.input_shape)
    RoI_input = Input(shape=C.input_shape)



    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training Fast R-CNN detector')

    pinfo('Parameters are set inside the script')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    lambdas = [1, 100]
    model_name = 'detector_mc_00'
    record_name = 'detector_mc_record_00'

    C.set_detector_record(model_name, record_name)
    C = detector_train(C)
