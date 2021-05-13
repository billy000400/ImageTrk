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

test_dp_list = ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
        "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art"]

def make_data_from_distribution()
