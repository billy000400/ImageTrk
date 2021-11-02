### Make Trianing set for the Fast-RCNN detector

import sys
from pathlib import Path
import shutil
import timeit
import pickle
import random
from copy import copy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Geometry import iou
from Information import *

import tensorflow as tf

def make_data(C,):

    # initialize path objects
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
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

    # calculate how many negative examples we want
    negThreshold = np.int(C.roiNum*C.negativeRate)

    file_idx = 0
    for img_idx, img in enumerate(imgNames):

        # register memory for input and output data
        rois = np.zeros(shape=(C.roiNum, 4))
        rois[:] = np.nan
        outputs_classifier = np.zeros(shape=(C.roiNum, len(oneHotEncoder)), dtype=np.float32)
        outputs_classifier[:] = np.nan
        outputs_regressor = np.zeros(shape=(C.roiNum, len(oneHotEncoder)*4), dtype=np.float32)
        outputs_regressor[:] = np.nan

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

        pos_tuples = []
        neg_tuples = []

        # iterate over proposed bboxes to sort them into positives and negatives
        for proposal in proposals:
            iou_highest=0
            label = None
            ref_bbox = None
            for bbox, pdgId in bbox_pdgId_pairs:
                iou_tmp = iou(bbox, proposal)
                if iou_tmp > iou_highest:
                    iou_highest = iou_tmp
                    label = pdgId
                    ref_bbox = bbox
            if label == '-11':
                continue
            if iou_highest > 0.5:
                pos_tuples.append((proposal, label, ref_bbox))
            elif iou_highest > 0.0:
                neg_tuples.append((proposal, 'bg', ref_bbox))

        # calculate the number of positive example and negative example
        posNum = len(pos_tuples)
        referenceNum = len(df_r_slice['ClassName'].tolist())

        if posNum < referenceNum:
            sys.stdout.write('\n')
            sys.stdout.flush()
            perr(f'Not enough good proposal for {img}')
            perr(f'Details: proposed: {posNum}, reference: {referenceNum}')
            sys.exit()

        negNum = len(neg_tuples)
        totNum = posNum + negNum

        roiNum = C.roiNum

        if totNum < roiNum:
            tuples_combined = pos_tuples+neg_tuples # The original/whole sample
            sampleNum = len(tuples_combined)

            tuples_selected = copy(tuples_combined)
            totNum = len(tuples_selected)
            roiNeedNum = roiNum-totNum

            while roiNeedNum != 0:

                if sampleNum < roiNeedNum:
                    tuples_selected += tuples_combined
                    totNum = len(tuples_selected)
                    roiNeedNum = roiNum - totNum
                else:
                    tuples_selected += random.sample(tuples_combined, roiNeedNum)
                    totNum = len(tuples_selected)
                    roiNeedNum = roiNum - totNum

            assert len(tuples_selected)==roiNum, pdebug(len(tuples_selected))

        else:

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
            pos_selected = random.sample(pos_tuples, posWant)
            neg_selected = random.sample(neg_tuples, negWant)

            # combine negative examples and positive examples and shuffle
            tuples_selected = pos_selected + neg_selected

        random.shuffle(tuples_selected)

        # copy the result to the registered memory
        for i, tuple in enumerate(tuples_selected):
            proposal, label, ref_bbox = tuple
            # proposal t = (x, y, w, h) as indicated in the original paper
            # (x,y) is the left upper corner
            t = [ proposal[0], proposal[3],\
                        (proposal[1]-proposal[0]), (proposal[3]-proposal[2]) ]
            rois[i] = np.array(t, dtype=np.float32)

            oneHotVector = oneHotEncoder[label]
            outputs_classifier[i] = oneHotVector

            # refernece bbox v = (x,y,w,h) as indicated in the original paper
            # (x,y) is the left upper corner
            v = [ ref_bbox[0], ref_bbox[3],\
                        (ref_bbox[1]-ref_bbox[0]), (ref_bbox[3]-ref_bbox[2]) ]
            record_start = np.where(oneHotVector==1)[0][0] *4
            record_end = record_start + 4
            outputs_regressor[i][record_start:record_end] = v

        roi_file = roi_dir.joinpath(f'roi_{str(file_idx).zfill(7)}.npy')
        Y_classifier_file = Y_classifier_dir.joinpath(f'y_classifier_{str(file_idx).zfill(7)}.npy')
        Y_regressor_file = Y_regressor_dir.joinpath(f'y_regressor_{str(file_idx).zfill(7)}.npy')

        # save data to disk
        np.save(roi_file, rois)
        np.save(Y_classifier_file, outputs_classifier)
        np.save(Y_regressor_file, outputs_regressor)

        file_idx += 1


    # save file path to config and dump it
    C.set_oneHotEncoder(oneHotEncoder)
    C.set_detector_training_data(roi_dir, Y_classifier_dir, Y_regressor_dir)


    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Making data for the Fast-RCNN detector')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    roiNum = 128
    negativeRate = 0.5

    C.set_roi_parameters(roiNum, negativeRate)

    C = make_data(C)

    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
