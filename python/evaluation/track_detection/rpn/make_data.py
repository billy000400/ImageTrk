import sys
from pathlib import Path
import shutil
import pickle
from collections import Counter
from math import sqrt

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate

from sqlalchemy import *

script_dir = Path.cwd().parent.parent.parent.joinpath('frcnn_mc_train')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Architectures import VGG16
from Layers import *
from Geometry import iou
from Information import *


from rpn_data_make_train import make_data, make_data_from_distribution

track_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/tracks'
data_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/data'
window = 3000 # unit: number of windows
resolution = 512
mode = 'normal'
mean = 5
std = 2

track_dir = Path(track_dir_str)
data_dir = Path(data_dir_str)
C = frcnn_config(track_dir, data_dir)
C.set_distribution(mean, std)
C.set_window(window)
C.set_resolution(resolution)
C = make_data(C, mode)

from rpn_data_make_validation import make_data, make_data_from_distribution
C = make_data(C, mode)

from rpn_data_preprocess_train_vgg16 import preprocess
# configure important parameters
anchor_scales = [0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
anchor_ratios = [[1,1],\
                    [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]
lower_limit = 0.3
upper_limit = 0.7

posCut = 32
nWant = 64

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))
vgg16  = VGG16()
C.set_base_net(vgg16)
C.set_anchor(anchor_scales, anchor_ratios)
C.set_label_limit(lower_limit, upper_limit)
C.set_sample_parameters(posCut, nWant)
C = preprocess(C)


from rpn_data_preprocess_validation_vgg16 import preprocess
C = preprocess(C)
