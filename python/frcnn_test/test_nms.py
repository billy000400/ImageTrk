### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.image import non_max_suppression_with_scores

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_config import Config
from frcnn_util import make_anchors, normalize_anchor, propose_score_bbox_list
from frcnn_rpn import rpn
from mu2e_output import *

from rpn_predict import predict
from rpn_test import region_proposal_analysis
### imports end

script, Nt = sys.argv
Nt = float(Nt)

pbanner()
psystem('Faster R-CNN Object Detection System')
pmode('Testing')

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

Ot_s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

IoU_standards = [0.5]

cwd = Path.cwd()
data_dir = cwd.parent.parent.joinpath('data')
csv_file = data_dir.joinpath("nms_0.1_result.csv")
if csv_file.is_file():
    df = pd.read_csv(csv_file, index_col=None)
else:
    df = pd.DataFrame(columns=["IoU_threshold", "Score_threshold", "Sigma", "Precision", "Recall", "Degeneracy"])



for Ot in Ot_s:
    for Sigma in Sigmas:
        tmp = {}
        tmp["IoU_threshold"] = Nt
        tmp["Score_threshold"] = Ot
        tmp["Sigma"] = Sigma
        predict(C, Nt, Ot, Sigma)
        df_tmp = region_proposal_analysis(C, IoU_standards)
        tmp["Precision"] = df_tmp.loc[0,0.5]
        tmp["Recall"] = df_tmp.loc[1,0.5]
        tmp["Degeneracy"] = df_tmp.loc[2,0.5]
        df = df.append(tmp, ignore_index=True)
        df.to_csv(csv_file, index=False)

print(df)
