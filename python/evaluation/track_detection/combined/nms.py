# Faster RCNN
# Filter detector output via nms

import sys
from pathlib import Path
import pickle
import timeit
from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores


script_dir = Path.cwd().parent.parent.parent.joinpath('frcnn_mc_train')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config

# Load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

# setup convenient directories
prediction_dir = cwd.joinpath('predictions')
performance_dir = cwd.joinpath('performances')

# Load frcnn prediction
ref_df = pd.read_csv(C.train_bbox_reference_file, index_col=0)
pred_df = pd.read_csv(prediction_dir.joinpath('rpn+detector_cls_predictions.csv'), index_col=0)

def detector_to_nms(ref_df, pred_df, nms_fn):
    imgs = ref_df['FileName'].unique().tolist()
    bbox_idx = 0
    dict_for_df={}
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
        selected_indices, selected_scores =nms_fn(pred_bboxes_raw_tf, scores_tf)

        selected_indices_list = selected_indices.numpy().tolist()
        bboxes = [ pred_bboxes_raw[index] for index in selected_indices_list ]
        scores = selected_scores.numpy().tolist()

        for score, bbox in zip(scores, bboxes):
            dict_for_df[bbox_idx] = {'FileName': str(img),\
                                'XMin':bbox[0],\
                                'XMax':bbox[1],\
                                'YMin':bbox[2],\
                                'YMax':bbox[3],\
                                'Score':score}
            bbox_idx += 1
    output_df = pd.DataFrame.from_dict(dict_for_df, "index")
    return output_df

def define_nms_fn(max_output_size, iou_threshold, score_threshold, soft_nms_sigma):
    def nms_fn(bboxes, scores):
        return non_max_suppression_with_scores(bboxes, scores,\
                    max_output_size=max_output_size,\
                    iou_threshold=iou_threshold, score_threshold=score_threshold,\
                    soft_nms_sigma=soft_nms_sigma)
    return nms_fn

# for iou in [0.45, 0.5, 0.55]:
#     nms_fn = define_nms_fn(max_output_size=3000,\
#         iou_threshold=iou, score_threshold=0.0,\
#         soft_nms_sigma=0.0)
#
#     df = detector_to_nms(ref_df, pred_df, nms_fn)
#
#
#     output_file = prediction_dir.joinpath(f"rpn+detector_cls+nms@{iou}_predictions.csv")
#     df.to_csv(output_file)

nms_fn = define_nms_fn(max_output_size=3000,\
    iou_threshold=0.6, score_threshold=0.5,\
    soft_nms_sigma=1.0)

df = detector_to_nms(ref_df, pred_df, nms_fn)


output_file = prediction_dir.joinpath(f"rpn+detector_cls+nms@IT=0.6_ST=0.5_Sigma=1.0_predictions.csv")
df.to_csv(output_file)
