### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import *
from Geomoetry import iou
from Information import *
from Configuration import frcnn_config

from mean_average_precision import MetricBuilder
### import ends

### This function analysis the result of NMS given a SINGLE set of parameters
def region_proposal_analysis(C, max_output_size, iou_threshold, score_threshold,\
 soft_nms_sigma, IoU_cuts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

    ### load reference and prediction bbox table to pandas framework
    # construct file object
    data_dir = C.sub_data_dir
    data_dir = data_dir.joinpath('NMS analysis nodes')
    data_dir.mkdir(exist_ok=True)
    reference_file = data_dir.joinpath(C.validation_bbox_reference_file)
    prediction_file = data_dir.joinpath(C.validation_bbox_proposal_file)


    # read csv to df
    ref_df = pd.read_csv(reference_file, index_col=None)
    pred_df = pd.read_csv(prediction_file, index_col=None)

    ### preparing the result dataframe
    columns = ['Metric/IoU_cuts']+IoU_cuts
    result_df = pd.DataFrame(columns = columns)
    precision_row = ['precision']
    recall_row = ['recall']
    degeneracy_row = ['degeneracy']
    map1_row = ['mAP@.75']
    map2_row = ['mAP@.5']
    map3_row = ['mAP@[.5,.95]']

    ### Grading the RPN prediction after NMS
    grading_grids = []
    imgs = ref_df['FileName'].unique()

    map1s, map2s, map3s = [], [], []

    for img_idx, img in enumerate(imgs):
        sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgs)}", '\r'))
        if img_idx+1 == len(imgs):
            sys.stdout.write('\n')
        sys.stdout.flush()





        ref_slice = ref_df[ref_df['FileName']==img]
        pred_slice = pred_df[pred_df['FileName']==img]

        ref_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax']] for index, row in ref_slice.iterrows() ]
        pred_bboxes_raw = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax']] for index, row in pred_slice.iterrows() ]

        # Preprocess for NMS
        scores = [ row['Score'] for index, row in pred_slice.iterrows() ]
        scores_tf = tf.constant(scores, dtype=tf.float32)
        pred_bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in pred_bboxes_raw ]
        pred_bboxes_raw_tf = tf.constant(pred_bboxes_raw_tf, dtype=tf.float32)

        # NMS to reduce duplicity
        selected_indices, selected_score =\
        non_max_suppression_with_scores(pred_bboxes_raw_tf, scores_tf,\
                    max_output_size=max_output_size,\
                    iou_threshold=iou_threshold, score_threshold=score_threshold,\
                    soft_nms_sigma=soft_nms_sigma)

        selected_indices_list = selected_indices.numpy().tolist()
        pred_bboxes = [ pred_bboxes_raw[index] for index in selected_indices_list ]

        selected_score = selected_score.numpy().tolist()

        gt = np.array([ [r['XMin'], r['YMin'], r['XMax'], r['YMax'], 0, 0, 0] for index,r in ref_slice.iterrows()])

        preds = np.array([ [b[0], b[2], b[1], b[3], 0, score]\
                    for b, score in zip(pred_bboxes, selected_score)])

        if len(pred_bboxes)==0:
            map1s.append(0)
            map2s.append(0)
            map3s.append(0)
        else:
            # create metric_fn
            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
            # add some samples to evaluation
            metric_fn.add(preds, gt)

            map1 = metric_fn.value(iou_thresholds=0.75)['mAP']
            map2 = metric_fn.value(iou_thresholds=0.5)['mAP']
            map3 = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))['mAP']

            map1s.append(map1)
            map2s.append(map2)
            map3s.append(map3)

        # calculate precision, recall, and degeneracy
        iNum = len(ref_bboxes)
        jNum = len(pred_bboxes)
        grading_grid = np.zeros(shape=(iNum, jNum))

        for i, j in np.ndindex(iNum, jNum):
            grading_grid[i,j] = iou(ref_bboxes[i], pred_bboxes[j])

        grading_grids.append(grading_grid)


    for iou_idx, iou_limit in enumerate(IoU_cuts):

        sys.stdout.write(t_info(f"Processing iou cuts: {iou_idx+1}/{len(IoU_cuts)}", '\r'))
        if iou_idx+1 == len(IoU_cuts):
            sys.stdout.write('\n')
        sys.stdout.flush()

        precisions = []
        recalls = []
        degeneracies = []

        for grading_grid in grading_grids:
            iNum, jNum = grading_grid.shape

            tp = 0
            fp = 0
            fn = 0

            for i in range(iNum):
                row = grading_grid[i]
                if np.any(row>=iou_limit):
                    degeneracy = np.count_nonzero(row>=iou_limit)
                    degeneracies.append(degeneracy)

                else:
                    degeneracies.append(0)
                    fn += 1

            for j in range(jNum):
                col = grading_grid[:,j]
                if np.any(col>=iou_limit):
                    tp += 1
                else:
                    fp += 1

            dom1 = tp+fp

            if dom1!=0:
                precision = tp/dom1
                precisions.append(precision)

            recall = tp/(tp+fn)
            recalls.append(recall)

        if len(precisions)!=0:
            precision_entry = np.array(precisions).mean()
            precision_row.append(precision_entry)
        else:
            precision_row.append([])

        if len(recalls)!=0:
            recall_entry = np.array(recalls).mean()
            recall_row.append(recall_entry)
        else:
            recall_row.append([])

        if len(degeneracies)!=0:
            degeneracy_entry = np.array(degeneracies).mean()
            degeneracy_row.append(degeneracy_entry)
        else:
            degeneracy_row.append([])

        map1_row.append(np.array(map1s).mean())
        map2_row.append(np.array(map2s).mean())
        map3_row.append(np.array(map3s).mean())


    precision_df = pd.DataFrame([precision_row], columns=columns)
    recall_df = pd.DataFrame([recall_row], columns=columns)
    degeneracy_df = pd.DataFrame([degeneracy_row], columns=columns)
    map1_df = pd.DataFrame([map1_row], columns=columns)
    map2_df = pd.DataFrame([map2_row], columns=columns)
    map3_df = pd.DataFrame([map3_row], columns=columns)

    result_df = result_df.append(precision_df, ignore_index=True)
    result_df = result_df.append(recall_df, ignore_index=True)
    result_df = result_df.append(degeneracy_df, ignore_index=True)
    result_df = result_df.append(map1_df, ignore_index=True)
    result_df = result_df.append(map2_df, ignore_index=True)
    result_df = result_df.append(map3_df, ignore_index=True)

    result_file = data_dir.joinpath(f'NMS_analysis_IT={iou_threshold}_ST={score_threshold}_Sigma={soft_nms_sigma}')
    result_df.to_csv(result_file)

    return result_df



if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Testing')


    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    max_output_size = 500
    iou_threshold = 1.0
    score_threshold = 0.0
    soft_nms_sigma = 0.0


    result_df = region_proposal_analysis(C, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
    print(result_df)
