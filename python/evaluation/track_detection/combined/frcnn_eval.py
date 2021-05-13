### imports
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import *
from Geometry import iou
from Information import *
from Configuration import frcnn_config

from mean_average_precision import MetricBuilder

# Load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

### object detection analysis function
# return common metrics: precision, recall, degeneracy, and mAPs
def od_analysis(ref_df, pred_df, IoU_cuts):
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
        pred_bboxes = [ [row['XMin'], row['XMax'], row['YMin'], row['YMax'], row['Score']] for index, row in pred_slice.iterrows() ]



        gt = np.array([ [r['XMin'], r['YMin'], r['XMax'], r['YMax'], 0, 0, 0] for index,r in ref_slice.iterrows()])
        preds = np.array([ [b[0], b[2], b[1], b[3], 0, b[4]]\
                    for b in pred_bboxes])

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

            # dom1 == 0 when no ROI is proposed
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
    return result_df

# load real bboxes and predicted bboxes
IoU_cuts = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ref_df = pd.read_csv(C.validation_bbox_proposal_file, index_col=0)




# pred1_df = pd.read_csv('rpn+detector_cls_prediction.csv', index_col=0)
# rs1_df = od_analysis(ref_df, pred1_df, IoU_cuts)
# rs1_file = Path.cwd().joinpath('rpn+detector_cls_result.csv')
# rs1_df.to_csv(rs1_file)
#
#
# pred2_df = pd.read_csv('rpn+detector_prediction.csv', index_col=0)
# rs2_df = od_analysis(ref_df, pred2_df, IoU_cuts)
# rs2_file = Path.cwd().joinpath('rpn+detector_result.csv')
# rs2_df.to_csv(rs2_file)

pred3_df = pd.read_csv('rpn+detector_cls+nms@0.6_prediction.csv', index_col=0)
rs3_df = od_analysis(ref_df, pred3_df, IoU_cuts)
rs3_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.6_result.csv')
rs3_df.to_csv(rs3_file)
print(rs3_df)

pred3_df = pd.read_csv('rpn+detector_cls+nms@0.5_prediction.csv', index_col=0)
rs3_df = od_analysis(ref_df, pred3_df, IoU_cuts)
rs3_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.5_result.csv')
rs3_df.to_csv(rs3_file)
print(rs3_df)

pred3_df = pd.read_csv('rpn+detector_cls+nms@0.4_prediction.csv', index_col=0)
rs3_df = od_analysis(ref_df, pred3_df, IoU_cuts)
rs3_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.4_result.csv')
rs3_df.to_csv(rs3_file)
print(rs3_df)
