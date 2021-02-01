import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from abstract import *
from mu2e_output import *
from frcnn_config import Config

def anchor_statistics(C):

    # unpack parameters from configuration

    # make parameters
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)
    anchors_arr = np.array(anchors)

    img_bbox_dict = make_img_bbox_dict(C.img_dir, C.bbox_file)
    img_bbox_list = [ [img_name, bbox_list] for img_name, bbox_list in img_bbox_dict.items() ]
    all_img_name = [img_name for img_name, bbox_list in img_bbox_list]

    bbox_Num = len(pd.read_csv(C.bbox_file, index_col=0).index)
    bbox_idx = 0

    all_track_number = []
    all_positive_anchor_number = []
    all_negative_anchor_number = []


    for img_name, bbox_list in img_bbox_list:

        all_track_number.append(len(bbox_list))

        score_bbox_map = make_score_bbox_map(anchors)
        for bbox in bbox_list:
            bbox_idx += 1
            sys.stdout.write(t_info(f'Scoring and labeling anchors by bbox: {bbox_idx}/{bbox_Num}', special='\r'))
            if bbox_idx == bbox_Num:
                sys.stdout.write('\n')
            sys.stdout.flush()
            score_bbox_map = update_score_bbox_map(score_bbox_map, bbox, anchors)

        score_map = [pyramid[0] for row in score_bbox_map for column in row for pyramid in column]
        score_map = np.array(score_map)
        positive_anchor_number = np.count_nonzero(score_map>=0.5)
        negative_anchor_number = score_map.size-positive_anchor_number
        all_positive_anchor_number.append(positive_anchor_number)
        all_negative_anchor_number.append(negative_anchor_number)

    all_img_name = [idx for idx, name in enumerate(all_img_name)]
    fig, ax1 = plt.subplots()
    ax1.plot(all_img_name, all_track_number, label='bbox number')
    ax1.plot(all_img_name, all_positive_anchor_number, label='raw positive anchor number')
    ax1.plot(all_img_name, all_negative_anchor_number, label='raw negative anchor number')

    if cwd.parent.parent.joinpath('tmp').joinpath('label_maps.npy').is_file():
        sampled_label_maps = np.load(cwd.parent.parent.joinpath('tmp').joinpath('label_maps.npy'))
        sampled_pos_anchor_num = [np.count_nonzero(label_map[~np.isnan(label_map)]==1) for label_map in sampled_label_maps]
        sampled_neg_anchor_num = [np.count_nonzero(label_map[~np.isnan(label_map)]==0) for label_map in sampled_label_maps]
        sampled_pos_anchor_num = np.array(sampled_pos_anchor_num)
        sampled_neg_anchor_num = np.array(sampled_neg_anchor_num)
        ax1.plot(all_img_name, sampled_pos_anchor_num, label='sampled positive anchor number')
        ax1.plot(all_img_name, sampled_neg_anchor_num, label='sampled negative anchor number')

    ax1.set_title('Anchor Number and BBox Number of Images')
    ax1.set(xlabel='Image index', ylabel='Number')
    ax1.legend()
    plt.show()



if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    anchor_statistics(C)
