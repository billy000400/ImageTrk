### imports
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from mean_average_precision import MetricBuilder

### object detection analysis function
# return common metrics: precision, recall, and APs
def od_analysis(ref_df, pred_df, IoU_cuts):
    ### preparing the result dataframe

    columns = ['Metric/IoU_cuts']+IoU_cuts
    result_df = pd.DataFrame(columns = columns)
    precision_row = ['precision']
    recall_row = ['recall']
    ap_row = ['AP']


    grading_grids = []
    imgs = ref_df['FileName'].unique()
    precision_cols = [ [] for i in IoU_cuts ]
    recall_cols = [ [] for i in IoU_cuts]
    ap_cols = [ [] for i in IoU_cuts] # every sub [] contains aps for every IoU

    for img_idx, img in enumerate(imgs):
        sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgs)}", '\r'))
        if img_idx+1 == len(imgs):
            sys.stdout.write('\n')
        sys.stdout.flush()

        ref_slice = ref_df[ref_df['FileName']==img]
        pred_slice = pred_df[pred_df['FileName']==img]

        ref_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax']] for index, row in ref_slice.iterrows() ]
        pred_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax'], row['Score']] for index, row in pred_slice.iterrows() ]



        gt = np.array([ [r['XMin'], r['YMin'], r['XMax'], r['YMax'], 0, 0, 0] for index,r in ref_slice.iterrows()])
        gt *= 512.0
        preds = np.array([ [b[0], b[2], b[1], b[3], 0, b[4]]\
                    for b in pred_bboxes])
        preds *= 512.0

        if len(pred_bboxes)==0:
            for ap_col in ap_cols:
                ap_col.append(0)
        else:
            # create metric_fn
            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
            # add some samples to evaluation
            metric_fn.add(preds, gt)

            pascal_val = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP'])
            coco_val= metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP'])

            for col_idx, iou_val in enumerate(IoU_cuts):

                ap = result[iou_val][0]['ap']
                final_precision = result[iou_val][0]['precision'][-1]
                final_recall = result[iou_val][0]['recall'][-1]

                precision_cols[col_idx].append(final_precision)
                recall_cols[col_idx].append(final_recall)
                ap_cols[col_idx].append(ap)

    precision_df = pd.DataFrame([precision_row], columns=columns)
    recall_df = pd.DataFrame([recall_row], columns=columns)
    ap_df = pd.DataFrame([ap_row], columns=columns)

    result_df = result_df.append(precision_df, ignore_index=True)
    result_df = result_df.append(recall_df, ignore_index=True)
    result_df = result_df.append(ap_df, ignore_index=True)

    return result_df
