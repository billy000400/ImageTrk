import sys
from pathlib import Path
import shutil
import timeit
import pickle
import random
from copy import copy, deepcopy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Geometry import iou
from Information import *

def check_data(C, checkNum):

    df = pd.read_csv(C.train_bbox_reference_file,index_col=None)
    imgNames = df['FileName'].unique().tolist()
    imgNum = len(imgNames)
    res = C.resolution

    roi_files = [c for c in C.train_rois.iterdir()]
    cls_files = [c for c in C.detector_train_Y_classifier.iterdir()]
    rgr_files = [c for c in C.detector_train_Y_regressor.iterdir()]

    for x in range(checkNum):

        idx = random.randint(0,imgNum-1)



        imgName = imgNames[idx]
        img = mpimg.imread(C.train_img_dir.joinpath(imgName))

        slice = df[df['FileName']==imgName]
        bboxes = [ [r['XMin'], r['XMax'], r['YMin'], r['YMax']] for z,r in slice.iterrows() ]




        rois = np.load(roi_files[idx])
        clss = np.load(cls_files[idx])
        rgrs = np.load(rgr_files[idx])

        for roi, cls, rgr in zip(rois, clss, rgrs):
            xmin, ymax, w, h = roi
            xmax = xmin+w
            ymin = ymax-h

            xmin = xmin*res
            xmax = xmax*res
            ymin, ymax = (1-ymax)*res, (1-ymin)*res

            proposal = deepcopy([xmin, xmax, ymin, ymax])

            w = xmax-xmin
            h = ymax-ymin

            xy = (xmin, ymin)

            if cls[0]==1:
                ec = 'k'
            else:
                ec = 'r'

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

            iou_max = 0
            for bbox in bboxes:
                xmin, xmax, ymin, ymax = bbox
                xmin = xmin*res
                xmax = xmax*res
                ymin, ymax = (1-ymax)*res, (1-ymin)*res
                rec_xy = (xmin, ymin)
                rec_width = abs(xmax-xmin)
                rec_height = abs(ymax-ymin)
                rect = Rectangle(rec_xy, rec_width, rec_height, linewidth=1, edgecolor='y', facecolor='none')
                ax1.add_patch(rect)

                ref_bbox = [xmin, xmax, ymin, ymax]
                iou_val = iou(proposal, ref_bbox)
                if iou_val > iou_max: iou_max=iou_val

            ax1.imshow(img)
            ax2.imshow(img)

            rect = Rectangle(xy, w, h, linewidth=1, edgecolor=ec, facecolor='none')
            ax2.add_patch(rect)

            pinfo(f'IoU: {iou_max}')
            plt.show()
            plt.close()

            if ec=='k':
                pinfo('Bad proposal. No adjustion suggested')
                continue
            else:
                pinfo('Good proposal. The proposal has been adjusted')
                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
                ax1.imshow(img)
                ax2.imshow(img)
                for bbox in bboxes:
                    xmin, xmax, ymin, ymax = bbox
                    xmin = xmin*res
                    xmax = xmax*res
                    ymin, ymax = (1-ymax)*res, (1-ymin)*res
                    rec_xy = (xmin, ymin)
                    rec_width = abs(xmax-xmin)
                    rec_height = abs(ymax-ymin)
                    rect = Rectangle(rec_xy, rec_width, rec_height, linewidth=1, edgecolor='y', facecolor='none')
                    ax1.add_patch(rect)

                xmin, ymax, w, h = rgr[4:]
                xmax = xmin+w
                ymin = ymax-h

                xmin = xmin*res
                xmax = xmax*res
                ymin, ymax = (1-ymax)*res, (1-ymin)*res

                w = xmax-xmin
                h = ymax-ymin

                xy = (xmin, ymin)
                rect = Rectangle(xy, w, h, linewidth=1, edgecolor='c', facecolor='none')
                ax2.add_patch(rect)
                plt.show()
                plt.close()







    return

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Checking data for the Fast-RCNN detector')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    checkNum = 5

    C = check_data(C, 5)
