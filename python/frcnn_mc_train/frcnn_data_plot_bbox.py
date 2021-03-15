import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *
from Config import frcnn_config as Config

def plot_bboxes(C):

    df = pd.read_csv(C.bbox_file, index_col=0)
    df = df[['FileName','XMin','XMax','YMin','YMax','ClassName']]

    img_names = df['FileName'].unique()
    for img_name in img_names:
        pinfo(img_name)
        img_path = C.img_dir.joinpath(img_name)
        img_path_str = str(img_path)

        fig, ax = plt.subplots(figsize=(6,6))
        img = cv2.imread(img_path_str)
        ax.imshow(img, interpolation='none', extent=[-810,810,-810,810])
        ax.set(xlabel='x(mm)', ylabel='y(mm)')

        # get rectangles
        slice = df[df['FileName']==img_name].to_numpy()
        for row in slice:
            xy = ((row[1]-0.5)*1620-10, (row[3]-0.5)*1620-10)
            width = (row[2]-row[1])*1620+10
            height = (row[4]-row[3])*1620+10
            rec = Rectangle(xy,width,height,fill=False,edgecolor='g')
            ax.add_patch(rec)
        plt.show()
        plt.close()


    return

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    plot_bboxes(C)
