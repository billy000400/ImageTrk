### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_util import *
from mu2e_output import *
from frcnn_config import Config
### import ends

def region_proposal_analysis(C, IoU_standards):

    ### load reference and prediction bbox table to pandas framework
    # construct file object
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    reference_file = data_dir.joinpath(C.test_bbox_table_reference)
    prediction_file = data_dir.joinpath(C.test_bbox_table_prediction)
    result_file = data_dir.joinpath('rpn_test_result.csv')

    # read csv to df
    ref_df = pd.read_csv(reference_file, index_col=None)
    pred_df = pd.read_csv(prediction_file, index_col=None)

    ### preparing the result dataframe
    columns = ['method']+IoU_standards
    result_df = pd.DataFrame(columns = columns)

    ### calculate average precision for each IoU standard
    # get img names
    imgs = ref_df['FileName'].unique()

    precision_row = ['precision']
    recall_row = ['recall']
    degeneracy_row = ['degeneracy']

    for iou_idx, iou_limit in enumerate(IoU_standards):

        sys.stdout.write(t_info(f"Processing iou standard: {iou_idx+1}/{len(IoU_standards)}\n"))
        sys.stdout.flush()

        precisions = []
        recalls = []
        degeneracies = []

        for img_idx, img in enumerate(imgs):
            sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgs)}", '\r'))
            if img_idx+1 == len(imgs):
                sys.stdout.write('\n')
            sys.stdout.flush()

            ref_slice = ref_df[ref_df['FileName']==img]
            pred_slice = pred_df[pred_df['FileName']==img]

            ref_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax']] for index, row in ref_slice.iterrows() ]
            pred_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax']] for index, row in pred_slice.iterrows() ]

            iNum = len(ref_bboxes)
            jNum = len(pred_bboxes)
            grading_grid = np.zeros(shape=(iNum, jNum))

            for i, j in np.ndindex(iNum, jNum):
                grading_grid[i,j] = iou(ref_bboxes[i], pred_bboxes[j])

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

            precision = tp/(tp+fp)
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

    print(result_df)

    result_df.to_csv(result_file)

    return result_df



if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Testing')

    IoU_standards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.test.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    region_proposal_analysis(C, IoU_standards)
