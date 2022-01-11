### This script is to visually compare the ground truth and the prediction
### made by WCNN
import sys
from pathlib import Path
import shutil
import timeit
import pickle
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event
from Abstract import *
from Architectures import WCNN

if __name__ == "__main__":
    track_dir = Path.cwd().parent.parent.parent.joinpath('tracks')
    dp_name = 'train_CeEndpoint-mix'
    dp_file = track_dir.joinpath(dp_name+'.db')
    gen = Event([dp_file], 10, eventNum=50)

    res = 256
    scales = [float(1/256)]
    weight_path = "wcnn_02_tanh.h5"
    wcnn = WCNN(resolution=256, anchor_scales=scales, weight_path=weight_path)
    testNum = 10

    for i in range(testNum):
        hits_all, tracks_all = gen.generate(mode='eval')
        map = hit2ztmap(hits_all, 256)
        windows, scores = wcnn.propose(map)

        print(scores)
