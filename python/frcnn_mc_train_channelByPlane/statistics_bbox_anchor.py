import sys
from pathlib import Path
import pickle
from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config

def bbox_statistics(C):
    input_shape = C.input_shape

    df = pd.read_csv(C.train_bbox_reference_file, index_col=0)
    df = df[['FileName','XMin','XMax','YMin','YMax','ClassName']]

    df = pd.eval('Width=(df.XMax-df.XMin)', target=df)
    df = pd.eval('Height=(df.YMax-df.YMin)', target=df)
    w_arr = df['Width'].to_numpy()
    h_arr = df['Height'].to_numpy()
    scale_arr = np.sqrt(w_arr * h_arr)
    scale_norm_arr = scale_arr
    scale_col = pd.DataFrame({'Scale':scale_arr})
    df = df.join(scale_col)
    df = pd.eval('Ratio=(df.Width / df.Height)',target=df)

    scale_arr = df['Scale'].to_numpy() * np.sqrt(input_shape[0] * input_shape[1])
    ratio_arr = df['Ratio'].to_numpy()
    ratio_arr[ratio_arr < 1] = 1/ratio_arr[ratio_arr < 1]

    # find closest x_center and y_center
    img_names = df['FileName'].unique()
    all_x_centers = []
    all_y_centers = []
    all_x_center_diff_avg = []
    all_y_center_diff_avg = []
    for img_name in img_names:
        slice = df[df['FileName']==img_name]
        x_centers = slice['XMax']-slice['XMin']
        x_centers = x_centers.to_numpy()
        if x_centers.size == 1:
            continue
        x_centers = np.sort(x_centers)
        all_x_centers.append(x_centers)
        x_center_diffs = x_centers[1:x_centers.shape[0]]-x_centers[0:x_centers.shape[0]-1]

        x_center_diff_avg = x_center_diffs.mean()
        all_x_center_diff_avg.append(x_center_diff_avg)

        y_centers = slice['YMax']-slice['YMin']
        y_centers = y_centers.to_numpy()
        y_centers = np.sort(y_centers)
        all_y_centers.append(y_centers)
        y_center_diffs = y_centers[1:y_centers.shape[0]]-y_centers[0:y_centers.shape[0]-1]
        y_center_diff_avg = y_center_diffs.mean()
        all_y_center_diff_avg.append(y_center_diff_avg)

    all_x_centers = np.array(all_x_centers)
    all_y_centers = np.array(all_y_centers)
    all_x_center_diff_avg = np.array(all_x_center_diff_avg)
    all_y_center_diff_avg = np.array(all_y_center_diff_avg)

    pinfo("bbox distance in x")
    pinfo(f"mean: {all_x_center_diff_avg.mean()}; min: {all_x_center_diff_avg.min()}")

    pinfo("bbox distance in y")
    pinfo(f"mean: {all_y_center_diff_avg.mean()}; min: {all_y_center_diff_avg.min()}")

    anchor_scales = [0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
    anchor_ratios = [[1,1],\
                        [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]

    segs_scale = [ ((scale, 0), (scale, 500)) for scale in anchor_scales]
    line_segments_scale = LineCollection(segs_scale, linewidth=2, colors='r',\
                    linestyle='solid', label='anchor scale')


    fig1, ax1 = plt.subplots()
    ax1.hist(scale_norm_arr, 100, histtype='step', label='real bbox scale')
    ax1.add_collection(line_segments_scale)
    ax1.set_title('BBox Normalized Effective Scale Histogram')
    ax1.set(xlabel='Normalized effective scale(0-to-1)', ylabel='Number of bboxes')
    plt.legend(loc="upper left")
    plt.show()

    segs_ratio = [((1,0),(1,800)), ((2,0),(2,800))]
    line_segments_ratio = LineCollection(segs_ratio, linewidth=2, colors='r',\
                    linestyle='solid', label='anchor ratio')


    fig2, ax2 = plt.subplots()
    ax2.hist(ratio_arr, 100, histtype='step', label='real bbox ratio')
    ax2.add_collection(line_segments_ratio)
    ax2.set_title('BBox Ratio Histogram')
    ax2.set(xlabel='Greater-than-one Ratio', xlim = [0,None], ylabel='Number of bboxes')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    bbox_statistics(C)
