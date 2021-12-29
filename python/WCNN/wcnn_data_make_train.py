import sys
from pathlib import Path
import shutil
import timeit
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event
from Configuration import wcnn_config

def make_data(C):
    track_dir = C.track_dir
    mean = C.trackNum_mean
    std = C.trackNum_std
    windowNum = C.window
    resolution = C.resolution

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 20

    ### Construct Path Objects
    dp_list = C.train_dp_list
    dp_name_iter = iter(dp_list)
    dp_name = next(dp_name_iter)
    db_file = track_dir.joinpath(dp_name+".db")

    data_dir = C.data_dir
    C.sub_data_dir = data_dir.joinpath(Path.cwd().name)
    data_dir = C.sub_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    ### directories for numpy data
    photographic_train_x_dir = data_dir.joinpath('photographic_large_train_X')
    photographic_train_y_dir = data_dir.joinpath('photographic_large_train_Y')

    ### directories for jpg data
    photo_train_in_dir = data_dir.joinpath('photographic_large_train_input_photo')
    photo_train_out_dir = data_dir.joinpath('photographic_large_train_output_truth')

    shutil.rmtree(photographic_train_x_dir, ignore_errors=True)
    shutil.rmtree(photographic_train_y_dir, ignore_errors=True)
    shutil.rmtree(photo_train_in_dir, ignore_errors=True)
    shutil.rmtree(photo_train_out_dir, ignore_errors=True)
    photographic_train_x_dir.mkdir(parents=True, exist_ok=True)
    photographic_train_y_dir.mkdir(parents=True, exist_ok=True)
    photo_train_in_dir.mkdir(parents=True, exist_ok=True)
    photo_train_out_dir.mkdir(parents=True, exist_ok=True)
    return

if __name__ == "__main__":
    pbanner()
    psystem('Window-based Convolutional Neural Network')
    pmode('Generating training data')
    pinfo('Input DType for testing: StrawHit')

    track_dir = Path.cwd().parent.parent.joinpath('tracks')
    data_dir = Path.cwd().parent.parent.joinpath('data')

    C = wcnn_config(track_dir, data_dir)

    dp_list = ['train_CeEndpoint-mix.db']
    resolution = 256

    C.set_train_dp_list(dp_list)
    C.set_resolution(resolution)

    start = timeit.default_timer()
    make_data(C)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')
