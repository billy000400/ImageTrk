import sys
import csv
from pathlib import Path
import shutil
from collections import Counter
import pickle
from copy import deepcopy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import *

import PIL
from PIL import Image

script_dir = Path.cwd().parent.parent.joinpath('photographic_mc_FCDenseNet')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config
from Abstract import binning_objects
from Database import *
from Information import *

from photographic_data_make_train import make_data_from_distribution, make_data
track_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/tracks'
data_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/data'

track_dir = Path(track_dir_str)
data_dir = Path(data_dir_str)

train_dp_list = ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000011.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000012.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000014.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000020.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000024.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000044.art"]

test_dp_list = ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art"]

C = extractor_config(track_dir, data_dir)
C.set_train_dp_list(train_dp_list)
C.set_val_dp_list(test_dp_list) # notice that we replace val with test here and you can test by using val

mode = 'normal'
window = 3000
mean = 5
std = 2
resolution = 256
C.set_distribution(mean, std)
C.set_window(window)
C.set_resolution(resolution)
C = make_data(C, mode)

from photographic_data_make_validation import make_data_from_distribution, make_data
C = make_data(C)

cwd = Path.cwd()
pickle_path = cwd.joinpath('extractor.test.config.pickle')
pickle.dump(C, open(pickle_path, 'wb'))
