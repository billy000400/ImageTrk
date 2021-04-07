import sys
import pickle
from pathlib import Path

import numpy as np
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
from Information import *

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))


df_r = pd.read_csv(C.validation_bbox_reference_file, index_col=None)
df_p = pd.read_csv(C.validation_bbox_proposal_file, index_col=None)

val_img_dir = C.validation_img_dir

imgNames = df_r['FileName'].unique().tolist()

threshold = 100

res = C.resolution

for index, imgName in enumerate(imgNames):
    slice_r = df_r[df_r['FileName']==imgName]
    slice_p = df_p[df_p['FileName']==imgName]
    if len(slice_p.index)>threshold:
        slice_p = slice_p.sort_values(by='Score', ascending=False).head(threshold)


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))


    img = mpimg.imread(val_img_dir.joinpath(imgName))
    height, width = img.shape[0:2]
    ax1.imshow(img)

    for i, bbox in slice_r.iterrows():
        xmin = bbox['XMin']
        xmax = bbox['XMax']
        ymin = bbox['YMin']
        ymax = bbox['YMax']

        xmin = xmin*res
        xmax = xmax*res
        ymin, ymax = (1-ymax)*res, (1-ymin)*res

        rec_xy = (xmin, ymin)
        rec_width = abs(xmax-xmin)
        rec_height = abs(ymax-ymin)
        rect = Rectangle(rec_xy, rec_width, rec_height, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

    ax2.imshow(img)

    for i, bbox in slice_p.iterrows():
        xmin = bbox['XMin']
        xmax = bbox['XMax']
        ymin = bbox['YMin']
        ymax = bbox['YMax']

        xmin = xmin*res
        xmax = xmax*res
        ymin, ymax = (1-ymax)*res, (1-ymin)*res

        rec_xy = (xmin, ymin)
        rec_width = abs(xmax-xmin)
        rec_height = abs(ymax-ymin)
        rect = Rectangle(rec_xy, rec_width, rec_height, linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

    plt.show()
