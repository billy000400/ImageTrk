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

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Abstract import *
from mu2e_output import *
from Config import frcnn_config as Config
### import ends

### This function analysis the result of NMS given a SINGLE set of parameters
def region_proposal_analysis(C, max_output_size, iou_threshold, score_threshold,\
 soft_nms_sigma, IoU_cuts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

    ### load reference and prediction bbox table to pandas framework
    # construct file object
    data_dir = C.img_dir.parent
    reference_file = data_dir.joinpath(C.bbox_file)
    prediction_file = data_dir.joinpath(C.bbox_prediction_file)


    # read csv to df
    ref_df = pd.read_csv(reference_file, index_col=None)
    pred_df = pd.read_csv(prediction_file, index_col=None)

    ### preparing the result dataframe
    columns = ['Metric/IoU_cuts']+IoU_cuts
    result_df = pd.DataFrame(columns = columns)
    precision_row = ['precision']
    recall_row = ['recall']
    degeneracy_row = ['degeneracy']

    ### Grading the RPN prediction after NMS
    grading_grids = []
    imgs = ref_df['FileName'].unique()
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
                    fn += 1

            for j in range(jNum):
                col = grading_grid[:,j]
                if np.any(col>=iou_limit):
                    tp += 1
                else:
                    fp += 1

            dom1 = tp+fp

            if dom1==0:
                precision = 0
            else:
                precision = tp/dom1

            recall = tp/(tp+fn)

            precisions.append(precision)
            recalls.append(recall)

        precision_entry = np.array(precisions).mean()
        recall_entry = np.array(recalls).mean()
        degeneracy_entry = np.array(degeneracies).mean()

        precision_row.append(precision_entry)
        recall_row.append(recall_entry)
        degeneracy_row.append(degeneracy_entry)

    precision_df = pd.DataFrame([precision_row], columns=columns)
    recall_df = pd.DataFrame([recall_row], columns=columns)
    degeneracy_df = pd.DataFrame([degeneracy_row], columns=columns)

    result_df = result_df.append(precision_df, ignore_index=True)
    result_df = result_df.append(recall_df, ignore_index=True)
    result_df = result_df.append(degeneracy_df, ignore_index=True)

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
