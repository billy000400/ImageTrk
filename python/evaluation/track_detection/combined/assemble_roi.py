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

from sqlalchemy import *

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Architectures import VGG16
from Layers import *
from Geometry import iou
from Information import *

def roi_to_detector(C):
    # initialize path objects
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    roi_dir = data_dir.joinpath('frcnn_test_X_RoIs')

    shutil.rmtree(roi_dir, ignore_errors=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # load reference and prediction dataframes
    df_r = pd.read_csv(C.train_bbox_reference_file, index_col=None)
    df_p = pd.read_csv(C.train_bbox_proposal_file, index_col=None)

    # one-hot encoder
    categories = df_r['ClassName'].unique().tolist()
    oneHotEncoder = {}
    # The first entry indicates if it is a negative example (background)
    oneHotEncoder['bg'] = np.identity(len(categories)+1)[0]
    for i, pdgId in enumerate(categories):
        oneHotEncoder[pdgId] = np.identity(len(categories)+1)[i+1]

    imgNames = df_r['FileName'].unique().tolist()
    assert imgNames==df_p['FileName'].unique().tolist(),\
        perr('bbox_file\'s images do not agree with bbox_prediction_file\'s images')

    file_idx = 0
    for img_idx, imgName in enumerate(imgNames):

        sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgNames)}", '\r'))
        if img_idx+1 == len(imgNames):
            sys.stdout.write('\n')
        sys.stdout.flush()

        roiNum = np.count_nonzero(df_p['FileName']==imgName)

        # register memory for input and output data
        rois = np.zeros(shape=(roiNum, 4))

        df_p_slice = df_p[df_p['FileName']==img]

        proposals = [[r['XMin'], r['XMax'], r['YMin'], r['YMax']]\
                    for i, r in df_p_slice.iterrows()]


        # copy the result to the registered memory
        for i, proposal in enumerate(proposals):
            proposal, label, ref_bbox = tuple
            # proposal t = (x, y, w, h) as indicated in the original paper
            # (x,y) is the left upper corner
            t = [ proposal[0], proposal[3],\
                        (proposal[1]-proposal[0]), (proposal[3]-proposal[2]) ]
            rois[i] = np.array(t, dtype=np.float32)

        roi_file = roi_dir.joinpath(f'roi_{str(file_idx).zfill(7)}.npy')

        # save data to disk
        np.save(roi_file, rois)

        file_idx += 1


    # save file path to config and dump it
    C.set_oneHotEncoder(oneHotEncoder)
    C.set_detector_training_data(roi_dir, None, None)


    return C
