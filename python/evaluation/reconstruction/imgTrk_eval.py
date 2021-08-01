import sys
from pathlib import Path
from scipy.stats import norm
import cv2
import pickle
import timeit
from datetime import datetime

import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, TimeDistributed, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import binning_objects
from Information import *
from Configuration import extractor_config
from Layers import *
from Architectures import FC_DenseNet
from HitGenerators import Stochastic

### Using a specific pair of CPU and GPU
# I pick the first GPU because it is faster
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.experimental.get_visible_devices())

from track_detection import define_nms_fn
from track_detection import track_detection
from track_extraction import track_extraction



### calculate the number of intersection
def overlap(list1, list2):
    counter = 0
    for x in list1:
        if x in list2:
            counter += 1
    return counter

### number of hits in the intersection over the number of hits in the union
# calculate iou for a tracking window
def iou(trk_r, trk_p):
    ious = []
    for ptcl_id in trk_r:
        hits_r = trk_r[ptcl_id]
        iou_value = 0
        for hits_p in trk_p:
            i = overlap(hits_r, hits_p)
            u = float(len(hits_p) + len(hits_r)-i)
            tmp_iou = i/u
            if tmp_iou > iou_value:
                iou_value = tmp_iou
        ious.append(iou_value)
    return ious

### number of hits in the intersection
# over the number of hits in the original track
def recall(trk_r, trk_p):
    recalls = []
    for ptcl_id in trk_r:
        hits_r = trk_r[ptcl_id]
        recall_value = 0
        for hits_p in trk_p:
            i = overlap(hits_r, hits_p)
            tmp_recall = i/float(len(hits_r))
            if tmp_recall > recall_value:
                recall_value = tmp_recall
        recalls.append(recall_value)
    return recalls

### number of hits in the intersection
# over the number of hits in the predicted track
def precision(trk_r, trk_p):
    precisions =[]

    for hits_p in trk_p:
        if len(hits_p)==0:
            continue
        precision_value = 0
        for ptcl_id in trk_r:
            hits_r = trk_r[ptcl_id]
            i = overlap(hits_r, hits_p)
            tmp_precision = i/float(len(hits_p))
            if tmp_precision > precision_value:
                precision_value = tmp_precision
        precisions.append(precision_value)
    return precisions


if __name__ == "__main__":

    metric_dir = Path.cwd().joinpath('metrics')
    metric_dir.mkdir(parents=True, exist_ok=True)

    track_dir = Path('/home/Billy/Mu2e/analysis/DLTracking/tracks')
    dp_list =  ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art"]

    db_list = [ track_dir.joinpath(dp+'.db') for dp in dp_list]

    mean = 5.0
    std = 2.0
    dist = norm(loc=mean, scale=1/std)

    gen = Stochastic(dist=dist, db_files=db_list, hitNumCut=20)
    windowNum = 1000

    nms = define_nms_fn(max_output_size=3000,\
            iou_threshold=.7, score_threshold=.5, soft_nms_sigma=1400.0)

    precision_values = []
    recall_values = []
    iou_values = []
    hitNum_values = []


    for i in range(windowNum):
        sys.stdout.write(t_info(f'Processing window: {i+1}/{windowNum}', '\r'))
        if i+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        hit_dict, trks_ref = gen.generate(mode='eval')
        bboxes = track_detection(hit_dict, nms)
        trks_pred = track_extraction(hit_dict, bboxes)
        trks_pred = [ trk for trk in trks_pred if len(trk)>5 ]

        precisions = precision(trks_ref, trks_pred)
        recalls = recall(trks_ref, trks_pred)
        ious = iou(trks_ref, trks_pred)
        hitNums = [len(trk_pred) for trk_pred in trks_pred]

        for precision_value in precisions:
            precision_values.append(precision_value)
        for recall_value in recalls:
            recall_values.append(recall_value)
        for iou_value in ious:
            iou_values.append(iou_value)
        for hitNum in hitNums:
            hitNum_values.append(hitNum)

    precision_path = metric_dir.joinpath('hit_precisions.list')
    pickle.dump(precision_values, open(precision_path, 'wb'))

    recall_path = metric_dir.joinpath('hit_recalls.list')
    pickle.dump(recall_values, open(recall_path, 'wb'))

    iou_path = metric_dir.joinpath('hit_ious.list')
    pickle.dump(iou_values, open(iou_path, 'wb'))

    hitNum_path = metric_dir.joinpath('hitNums.list')
    pickle.dump(hitNum_values, open(hitNum_path, 'wb'))
