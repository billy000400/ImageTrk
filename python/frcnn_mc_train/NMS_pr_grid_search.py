### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import frcnn_config
from Abstract import make_anchors, normalize_anchor, propose_score_bbox_list
from Layers import rpn
from Information import *

from NMS_pr_analysis import region_proposal_analysis
### imports end

### Using a specific pair of CPU and GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.experimental.get_visible_devices())

pbanner()
psystem('Faster R-CNN Object Detection System')
pmode('Testing')

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

max_output_size = 2000
ITs = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
STs = [0.5, 0.6, .7, .8, .9]
#STs = list(np.round(np.arange(0.99,0.998,0.001,dtype=np.float32),3))
# Sigmas = np.linspace(0,1,11).tolist()
Sigmas = [0.1, 1.0, 1e2, 1e3, 1e4]
IoU_cuts = [0.75]


data_dir = C.sub_data_dir
csv_file = data_dir.joinpath("NMS_grid_search_soft_IoU>0.75.csv")

df = pd.DataFrame(columns=["IoU_threshold", "Score_threshold", "Sigma", "Precision", "Recall", "Degeneracy", "mAP@.5", "mAP@.75", "mAP@[.5,.95]"])

prNum = len(ITs)*len(STs)*len(Sigmas)
prIdx = 0
for IT in ITs:
    for ST in STs:
        for Sigma in Sigmas:
            for iou in IoU_cuts:
                pinfo(f"Evaluating parameter set: {prIdx+1}/{prNum}")
                prIdx += 1
                tmp = {}
                tmp["IoU_threshold"] = IT
                tmp["Score_threshold"] = ST
                tmp["Sigma"] = Sigma
                df_tmp = region_proposal_analysis(C, max_output_size, IT, ST, Sigma, IoU_cuts=IoU_cuts)
                tmp['IoU_cut'] = iou
                tmp["Precision"] = df_tmp.loc[0,iou]
                tmp["Recall"] = df_tmp.loc[1,iou]
                tmp["Degeneracy"] = df_tmp.loc[2,iou]
                tmp['mAP@.75'] = df_tmp.loc[3,iou]
                tmp['mAP@.5'] = df_tmp.loc[4,iou]
                tmp['mAP@[.5,.95]'] = df_tmp.loc[5,iou]
                df = df.append(tmp, ignore_index=True)
                df.to_csv(csv_file, index=False)

# don't use pinfo, because you cannot concatenate string with a dataframe
print(df)
