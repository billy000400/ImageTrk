import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config
from Geometry import iou

pbanner()
psystem('Faster R-CNN Object Detection System')
pmode('Training')

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

df = pd.read_csv(C.train_bbox_reference_file, index_col=None)

imgNames = df['FileName'].unique().tolist()

maxIoUs = []

for imgName in imgNames:
    slice = df[df['FileName']==imgName]
    bboxes = [ [r['XMin'], r['XMax'], r['YMin'], r['YMax'] ] for index, r in slice.iterrows() ]

    bboxNum = len(bboxes)
    maxIoU = 0

    for i in range(bboxNum):
        bbox1 = bboxes[i]

        for j in range(1,bboxNum-i):
            bbox2 = bboxes[i+j]
            iou_val = iou(bbox1, bbox2)

            if iou_val > maxIoU:
                maxIoU = iou_val

    maxIoUs.append(maxIoU)

maxIoUs = np.array(maxIoUs, dtype=np.float32)

pinfo(f'Max IoU statistics: min: {maxIoUs.min()}, max: {maxIoUs.max()}, avg: {maxIoUs.mean()}')

plt.hist(maxIoUs, histtype='step')
plt.xlabel('Max IoU')
plt.ylabel('NUmber of images')
plt.title('Distribution of Max Cross-IoU of True BBoxes')
plt.show()
