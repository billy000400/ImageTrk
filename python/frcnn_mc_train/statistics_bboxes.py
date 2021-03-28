import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config

def bbox_statistics(C):
    input_shape = C.input_shape

    df = pd.read_csv(C.bbox_reference_file, index_col=0)
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

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.hist(all_x_center_diff_avg, 20)
    ax1.set_title('Average x Distance of Adjacent BBoxes Histogram')
    ax1.set(xlabel='Normalized x distance', ylabel='Number of bboxes')
    ax2.hist(all_y_center_diff_avg, 20)
    ax2.set_title('Average y Distance of Adjacent BBoxes Histogram')
    ax2.set(xlabel='Normalized y distance', ylabel='Number of bboxes')
    plt.tight_layout()
    plt.show()


    fig, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2,2)
    ax3.hist(scale_norm_arr, 100)
    ax3.set_title('BBox Normalized Effective Scale Histogram')
    ax3.set(xlabel='Normalized effective scale(0-to-1)', ylabel='Number of bboxes')

    ax4.hist(scale_arr, 100)
    ax4.set_title('BBox Effective Scale Histogram')
    ax4.set(xlabel='Effective scale(pixels)', ylabel='Number of bboxes')

    ax5.hist(ratio_arr, 100)
    ax5.set_title('BBox Ratio Histogram')
    ax5.set(xlabel='Greater-than-one Ratio', xlim = [0,None], ylabel='Number of bboxes')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    bbox_statistics(C)
