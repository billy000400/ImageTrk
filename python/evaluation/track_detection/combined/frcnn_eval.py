### imports
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import *
from Geometry import intersection, union, iou
from Information import *
from Configuration import frcnn_config

from mean_average_precision import MetricBuilder

# Load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

### setup convenient directories
prediction_dir = cwd.joinpath('predictions')
performance_dir = cwd.joinpath('performances')
performance_dir.mkdir(parents=True, exist_ok=True)

### object detection analysis function
# return common metrics: precision, recall, degeneracy, and mAPs
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

        pickle.dump(gt, open('gt', 'wb'))
        pickle.dump(preds, open('preds', 'wb'))

        recall_dict1 = {}

        if len(pred_bboxes)==0:
            for ap_col in ap_cols:
                ap_col.append(0)
        else:
            # create metric_fn
            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
            # add some samples to evaluation
            metric_fn.add(preds, gt)
            result = metric_fn.value(iou_thresholds=IoU_cuts, recall_thresholds=np.arange(0., 1.1, 0.1))

            for col_idx, iou_val in enumerate(IoU_cuts):

                ap = result[iou_val][0]['ap']
                final_precision = result[iou_val][0]['precision'][-1]
                final_recall = result[iou_val][0]['recall'][-1]

                precision_cols[col_idx].append(final_precision)
                recall_cols[col_idx].append(final_recall)
                ap_cols[col_idx].append(ap)


        # calculate precision, recall, and degeneracy
        # iNum = len(ref_bboxes)
        # jNum = len(pred_bboxes)
        # grading_grid = np.zeros(shape=(iNum, jNum))
        #
        # for i, j in np.ndindex(iNum, jNum):
        #     grading_grid[i,j] = iou(ref_bboxes[i], pred_bboxes[j])
        #
        # grading_grids.append(grading_grid)

    for precision_col, recall_col, ap_col in zip(precision_cols, recall_cols, ap_cols):
        precision_row.append(np.array(precision_col).mean())
        recall_row.append(np.array(recall_col).mean())
        ap_row.append(np.array(ap_col).mean())






    ### Grading the RPN prediction after NMS
    # for iou_idx, iou_limit in enumerate(IoU_cuts):
    #     sys.stdout.write(t_info(f"Processing iou cuts: {iou_idx+1}/{len(IoU_cuts)}", '\r'))
    #     if iou_idx+1 == len(IoU_cuts):
    #         sys.stdout.write('\n')
    #     sys.stdout.flush()
    #
    #
    #     precisions = []
    #     recalls = []
    #     degeneracies = []
    #
    #     for grading_grid in grading_grids:
    #         iNum, jNum = grading_grid.shape
    #
    #         tp = 0
    #         fp = 0
    #         fn = 0
    #
    #         for i in range(iNum):
    #             row = grading_grid[i]
    #             if np.any(row>=iou_limit):
    #                 degeneracy = np.count_nonzero(row>=iou_limit)
    #                 degeneracies.append(degeneracy)
    #
    #             else:
    #                 degeneracies.append(0)
    #                 fn += 1
    #
    #         for j in range(jNum):
    #             col = grading_grid[:,j]
    #             if np.any(col>=iou_limit):
    #                 tp += 1
    #             else:
    #                 fp += 1
    #
    #         dom1 = tp+fp
    #
    #         # dom1 == 0 when no ROI is proposed
    #         if dom1!=0:
    #             precision = tp/dom1
    #             precisions.append(precision)
    #
    #         recall = tp/(tp+fn)
    #         recalls.append(recall)
    #
    #     if len(precisions)!=0:
    #         precision_entry = np.array(precisions).mean()
    #         precision_row.append(precision_entry)
    #     else:
    #         precision_row.append([])
    #
    #     if len(recalls)!=0:
    #         recall_entry = np.array(recalls).mean()
    #         recall_row.append(recall_entry)
    #     else:
    #         recall_row.append([])
    #
    #     if len(degeneracies)!=0:
    #         degeneracy_entry = np.array(degeneracies).mean()
    #         degeneracy_row.append(degeneracy_entry)
    #     else:
    #         degeneracy_row.append([])


    precision_df = pd.DataFrame([precision_row], columns=columns)
    recall_df = pd.DataFrame([recall_row], columns=columns)
    ap_df = pd.DataFrame([ap_row], columns=columns)

    result_df = result_df.append(precision_df, ignore_index=True)
    result_df = result_df.append(recall_df, ignore_index=True)
    result_df = result_df.append(ap_df, ignore_index=True)

    return result_df

# load real bboxes and predicted bboxes
IoU_cuts = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ref_df = pd.read_csv(C.train_bbox_reference_file, index_col=0)

pred_files = [f for f in prediction_dir.glob('*.csv')]
for i, pred_file in enumerate(pred_files):
    pinfo(f'Evaluating predictions {i+1}/{len(pred_files)}: {pred_file.name}')
    pred_df = pd.read_csv(pred_file, index_col=0)
    fileName = pred_file.name[::-1].split('_',1)[1][::-1]+'_performance.csv'
    file = performance_dir.joinpath(fileName)

    if file.exists():
        read_df = pd.read_csv(file, index_col=0)
        print(read_df)
    else:
        rs_df = od_analysis(ref_df, pred_df, IoU_cuts)
        rs_df.to_csv(file)
        print(rs_df)

# pred_files = [f for f in prediction_dir.glob('*.csv')]
# for i, pred_file in enumerate(pred_files):
#     if i<5:
#         continue
#     pinfo(f'Evaluating predictions {i+1}/{len(pred_files)}: {pred_file.name}')
#     pred_df = pd.read_csv(pred_file, index_col=0)
#     fileName = pred_file.name[::-1].split('_',1)[1][::-1]+'_performance.csv'
#     file = performance_dir.joinpath(fileName)
#
#     rs_df = od_analysis(ref_df, pred_df, IoU_cuts)
#     print(rs_df)


# pred1_df = pd.read_csv('rpn+detector_cls_prediction.csv', index_col=0)
# rs1_df = od_analysis(ref_df, pred1_df, IoU_cuts)
# rs1_file = Path.cwd().joinpath('rpn+detector_cls_result.csv')
# rs1_df.to_csv(rs1_file)
# print('rpn+detector_cls')
# print(rs1_df)
#
#
# pred2_df = pd.read_csv('rpn+detector_prediction.csv', index_col=0)
# rs2_df = od_analysis(ref_df, pred2_df, IoU_cuts)
# rs2_file = Path.cwd().joinpath('rpn+detector_result.csv')
# rs2_df.to_csv(rs2_file)
# print('rpn+detector')
# print(rs2_df)
#
# pred3_df = pd.read_csv('rpn+detector_cls+nms@0.6_prediction.csv', index_col=0)
# rs3_df = od_analysis(ref_df, pred3_df, IoU_cuts)
# rs3_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.6_result.csv')
# rs3_df.to_csv(rs3_file)
# print('rpn+detector_cls+nms@0.6')
# print(rs3_df)
#
# pred4_df = pd.read_csv('rpn+detector_cls+nms@0.5_prediction.csv', index_col=0)
# rs4_df = od_analysis(ref_df, pred4_df, IoU_cuts)
# rs4_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.5_result.csv')
# rs4_df.to_csv(rs4_file)
# print('rpn+detector_cls+nms@0.5')
# print(rs4_df)
#
# pred5_df = pd.read_csv('rpn+detector_cls+nms@0.4_prediction.csv', index_col=0)
# rs5_df = od_analysis(ref_df, pred5_df, IoU_cuts)
# rs5_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.4_result.csv')
# rs5_df.to_csv(rs5_file)
# print('rpn+detector_cls+nms@0.4')
# print(rs5_df)
#
# pred6_df = pd.read_csv(C.validation_bbox_proposal_file, index_col=0)
# rs6_df = od_analysis(ref_df, pred6_df, IoU_cuts)
# rs6_file = Path.cwd().joinpath('rpn2000_result.csv')
# rs6_df.to_csv(rs6_file)
# print(rs6_df)
#
# pred7_df = pd.read_csv('rpn+detector_cls+nms@0.45_prediction.csv', index_col=0)
# rs7_df = od_analysis(ref_df, pred7_df, IoU_cuts)
# rs7_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.45_result.csv')
# rs7_df.to_csv(rs7_file)
# print('rpn+detector_cls+nms@0.45')
# print(rs7_df)
#
# pred8_df = pd.read_csv('rpn+detector_cls+nms@0.55_prediction.csv', index_col=0)
# rs8_df = od_analysis(ref_df, pred8_df, IoU_cuts)
# rs8_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.55_result.csv')
# rs8_df.to_csv(rs8_file)
# print('rpn+detector_cls+nms@0.55')
# print(rs8_df)
#
# pred9_df = pd.read_csv('rpn+detector_cls+nms@0.65_prediction.csv', index_col=0)
# rs9_df = od_analysis(ref_df, pred9_df, IoU_cuts)
# rs9_file = Path.cwd().joinpath('rpn+detector_cls+nms@0.65_result.csv')
# rs9_df.to_csv(rs9_file)
# print('rpn+detector_cls+nms@0.65')
# print(rs9_df)
