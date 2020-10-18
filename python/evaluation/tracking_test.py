# import starts
import sys
from pathlib import Path
import timeit
import csv

import numpy as np

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *
# import ends

data_dir = Path.cwd().parent.parent.joinpath('data')
reference_file = data_dir.joinpath('tracks_reference.csv')
prediction_file = data_dir.joinpath('tracks_prediction.csv')

pinfo("Loading csv documents")
with open(reference_file) as f1:
    f1_reader = csv.reader(f1)
    ref = list(f1_reader)

with open(prediction_file) as f2:
    f2_reader = csv.reader(f2)
    pred = list(f2_reader)

iNum = len(ref)
jNum = len(pred)

score_grid = np.zeros(shape=(iNum, jNum))

def iou_1D(list1, list2):
    intersection = 0
    union = len(list1)+len(list2)
    for e1 in list1:
        if e1 in list2:
            intersection +=1
    union = union - intersection
    return intersection/union


pinfo("Generating the score grid")
pinfo(f'iMax, jMax = {iNum}, {jNum}')
for i in range(iNum):
    for j in range(jNum):
        sys.stdout.write(t_info(f'Sliding grid ({i+1},{j+1})', special='\r'))
        if (i+1 == iNum) and (j+1 == jNum):
            sys.stdout.write('\n')
        sys.stdout.flush()
        t_ref = ref[i]
        t_pred = pred[j]
        iou = iou_1D(t_ref, t_pred)
        score_grid[i,j] = iou

pinfo("Evaluating")

standards = [0.1, 0.2, 0.3, 0.4, 0.5]

for iou_limit in standards:
    tp = 0
    fp = 0
    fn = 0

    degeneracies = []
    for i in range(iNum):
        row = score_grid[i]
        if np.any(row>=iou_limit):
            degeneracy = np.count_nonzero(row>=iou_limit)
        else:
            fn += 1

    for j in range(jNum):
        col = score_grid[:,j]
        if np.any(col>=iou_limit):
            tp += 1
        else:
            fp += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    degeneracy = np.array(degeneracy).mean()
    print(f"iou_threshold: {iou_limit}, precision: {precision}, recall: {recall}, degeneracy: {degeneracy}")
