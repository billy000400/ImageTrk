### Make Trianing set for the Fast-RCNN detector

import sys
from pathlib import Path
import shutil
import timeit
import pickle
import random

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from TrackDB_Classes import *
from Config import frcnn_config as Config
from Abstract import binning_objects
from Geometry import iou
from mu2e_output import *

import tensorflow as tf

def make_data(C, roiNum, negativeRate):

    # initialize path objects
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    data_dir.mkdir(parents=True, exist_ok=True)

    roi_dir = data_dir.joinpath('detector_train_X_RoIs')
    Y_classifier_dir = data_dir.joinpath('detector_train_Y_classifier')
    Y_regressor_dir = data_dir.joinpath('detector_train_Y_regressor')

    shutil.rmtree(roi_dir, ignore_errors=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(Y_classifier_dir, ignore_errors=True)
    Y_classifier_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(Y_regressor_dir, ignore_errors=True)
    Y_regressor_dir.mkdir(parents=True, exist_ok=True)

    roi_file = roi_dir.joinpath('detector_train_X_RoIs.npy')
    Y_classifier_file = Y_classifier_dir.joinpath('detector_train_Y_classifier.npy')
    Y_regressor_file = Y_regressor_dir.joinpath('detector_train_Y_regressor.npy')

    # load reference and prediction dataframes
    df_r = pd.read_csv(C.bbox_file, index_col=0)
    df_p = pd.read_csv(C.bbox_prediction_file, index_col=0)

    # one-hot encoder
    categories = df_r['ClassName'].unique().tolist()
    oneHotEncoder = {}
    # The first entry indicates if it is a negative example (background)
    oneHotEncoder['bg'] = np.identity(len(categories)+1)[0]
    for i, pdgId in enumerate(categories):
        oneHotEncoder[pdgId] = np.identity(len(categories)+1)[i+1]


    # register memory for input and output data
    inputs = np.load(C.inputs_npy)
    imgNum = inputs.shape[0]
    rois = np.zeros(shape=(imgNum, roiNum, 4))
    outputs_classifier = np.zeros(shape=(imgNum, roiNum, len(categories)+1), dtype=np.float32)
    outputs_regressor = np.zeros(shape=(imgNum, roiNum, 4), dtype=np.float32)

    imgNames = df_r['FileName'].unique().tolist()
    assert imgNames==df_p['FileName'].unique().tolist(),\
        perr('bbox_file\'s images do not agree with bbox_prediction_file\'s images')

    # calculate how many negative examples we want
    negThreshold = np.int(roiNum*negativeRate)

    for img_idx, img in enumerate(imgNames):

        sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgNames)}", '\r'))
        if img_idx+1 == len(imgNames):
            sys.stdout.write('\n')
        sys.stdout.flush()

        df_r_slice = df_r[df_r['FileName']==img]
        df_p_slice = df_p[df_p['FileName']==img]

        bbox_pdgId_pairs =\
         [   [[r['XMin'], r['XMax'], r['YMin'], r['YMax']],\
                        r['ClassName']] \
                 for i, r in df_r_slice.iterrows()]
        proposals = [ [r['XMin'], r['XMax'], r['YMin'], r['YMax']] for i, r in df_p_slice.iterrows()]

        positives = []
        negatives = []

        # iterate over proposed bboxes to sort them into positives and negatives
        for proposal in proposals:
            iou_highest=0
            label = None
            for bbox, pdgId in bbox_pdgId_pairs:
                iou_tmp = iou(bbox, proposal)
                if iou_tmp > iou_highest:
                    iou_highest = iou_tmp
                    label = pdgId
            if iou_highest > 0.5:
                positives.append([proposal, label])
            elif iou_highest > 0.1:
                negatives.append([proposal, 'bg'])

        # calculate the number of positive example and negative example
        posNum = len(positives)
        negNum = len(negatives)
        totNum = posNum + negNum

        assert totNum >= roiNum,\
            perr(f'Your RoI Number per image is {roiNum}, '
             f'but img {img_idx} only has {totNum} trainable RoIs.')

        if negNum < negThreshold:
            negWant = negNum
            posWant = roiNum - negWant
        elif (negThreshold + posNum) >= roiNum:
            negWant = negThreshold
            posWant = roiNum - negThreshold
        else:
            posWant = posNum
            negWant = roiNum - posWant

        # randomly select RoIs for training
        pos_selected = random.sample(positives, posWant)
        neg_selected = random.sample(negatives, negWant)

        # combine negative examples and positive examples and shuffle
        proposal_selected = pos_selected + neg_selected
        random.shuffle(proposal_selected)

        # copy the result to the registered memory
        for i, proposal in enumerate(proposal_selected):
            bbox_p, label = proposal
            outputs_classifier[img_idx][i] = oneHotEncoder[label]
            outputs_regressor[img_idx][i] = np.array(bbox_p, dtype=np.float32)

    # save data to disk
    np.save(Y_classifier_file, outputs_classifier)
    np.save(Y_regressor_file, outputs_regressor)

    # save file path to config and dump it
    C.set_detector_training_data(Y_classifier_file, Y_regressor_file)
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Making data for the Fast-RCNN detector')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    roiNum = 49
    negativeRate = 0.75

    make_data(C, roiNum, negativeRate)
