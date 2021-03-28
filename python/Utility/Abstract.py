## @package Abstract
#
# Abstract methods for processing and sorting data

from pathlib import Path
import sys
import math
import random
import numpy as np
import pandas as pd
from Geometry import *



## Bin objects into bin_array by features.
#
# \pr{objects, array like, Objects to be binned.}
# \pr{features, array like, Features by which objects will be binned. Every feature corresponds to an object at the same index.}
# \pr{bin_array, array like, A bin array that every element specifies a left or right boundary or a bin.}
# \rt{result, nested list, Every sub-list is a list of objects laying in the corresponding bin.}
#  Explicitly, the features of objects laying in a bin satisfie bin_left \f$\le\f$ feature < bin_right.
#  The first bin contains objects whose features are \f$-\infty\f$ < feature < first_bin_left.
def binning_objects(objects, features, bin_array):

    objs = objects
    ftrs = features
    bins = sorted(bin_array)

    try:
        objs = objs.to_list()
        ftrs = ftrs.to_list()
        bins = bins.to_list()
    except:
        pass

    assert len(objects)==len(features), \
        t_error('The lengths of object and feature are not equal')

    obj_ftr_raw  = dict(zip(objs,ftrs))
    obj_ftr = sorted( obj_ftr_raw.items(), key=lambda item: item[1])
    obj_ftr_it = iter(obj_ftr)

    result = []
    obj_ftr_idx = 0
    obj_ftr_idx_max = len(obj_ftr)-1

    for idx in range(len(bins)):
        bin_result = []

        if obj_ftr_idx == obj_ftr_idx_max+1:
            result.append(bin_result)
            continue

        if idx == 0:
            bin_left = float("-inf")
            bin_right = bins[0]
        else:
            bin_left = bins[idx-1]
            bin_right = bins[idx]

        while obj_ftr_idx <= obj_ftr_idx_max:
            if (obj_ftr[obj_ftr_idx][1] >= bin_left) and (obj_ftr[obj_ftr_idx][1] < bin_right):
                bin_result.append(obj_ftr[obj_ftr_idx][0])
                obj_ftr_idx += 1
            else:
                break

        result.append(bin_result)

    return result


## Normalize anchors' parameters from pixel presentations to [0,1] presentations.
#
# @param input_shape :<SPAN>&nbsp;&nbsp;&nbsp;&nbsp;list</SPAN> \n successful
# \pr{anchor, list, Unormalized list}
# [xmin, xmax, ymin, ymax].
# \pr{input_shape, list, [}
# row number, col number].
# \rt{normalized_anchor, list, A list of }
# [xmin, xmax, ymin, ymax] that every element is in [0,1]
def normalize_anchor(anchor, input_shape):

    width = input_shape[1]
    height = input_shape[0]
    xmin = anchor[0]/width
    xmax = anchor[1]/width
    ymin = anchor[2]/height
    ymax = anchor[3]/height
    return [xmin, xmax, ymin, ymax]

## Make an anchor pyramid at a given position
#
# \pr{anchor_ratios, nested list, Normalized anchor ratios. Every sublist}
# is of the form [width, height], where all elements are between 0 and 1.
# \pr{anchor_scales, list, Normalized anchor scales. Elements are between 0}
# and 1
# \pr{pos_center, list, Normalzied position of the pyramid's center.}
# \rt{pyramid, nested list, The returned anchor pyramid is a nested list}
# , in which every sub-list is an anchor represented by normalized
# [xmin, xmax, ymin, ymax]
def make_anchor_pyramid(anchor_scales, anchor_ratios, pos_center):

    x = pos_center[0]
    y = pos_center[1]
    num_ratios = len(anchor_ratios)
    num_anchors = len(anchor_scales) * num_ratios

    pyramid = np.zeros(shape=(num_anchors,4))
    for i1, scale in enumerate(anchor_scales):
        for i2, xy_set in enumerate(anchor_ratios):
            for j, side in enumerate(xy_set):
                if j == 0: # width
                    width = scale*side
                    xmin = x-width/2
                    xmax = x+width/2
                    pyramid[i1*num_ratios+i2][0]= xmin
                    pyramid[i1*num_ratios+i2][1]= xmax
                else: # height
                    height = scale*side
                    ymin = y-height/2
                    ymax = y+height/2
                    pyramid[i1*num_ratios+i2][2] = ymin
                    pyramid[i1*num_ratios+i2][3] = ymax

    return pyramid

## Make all anchors given an image
#
# \pr{input_shape, list or tuple, (img_height}
# , img_width).
# \pr{ratio, int, Average pixel distance between two adjacent anchors.}
# \pr{anchor_scales, list or tuple, Every element of anchor_scales is in [0}
# , 1], which represents the relative size to image's side length.
# \pr{anchor_ratios, nested list or tuple, Every sublist of anchor_ratios is}
# a 2-element list of the form [width, height], in which width and height satisfy
# height\f$ \times \f$width = 1.
# \rt{anchors, nested list, The first 2 indices represent the column and}
# row number of the anchor pyramid. The last index indicates a specific
# anchor in the anchor pyramid. Every anchor is a list of normalized
# [xmin, xmax, ymin, ymax]
def make_anchors(input_shape, ratio, anchor_scales, anchor_ratios):

    # Calculate the shape of the anchor map
    num_anchors = len(anchor_scales) * len(anchor_ratios)
    img_height = input_shape[0]
    img_width = input_shape[1]
    num_row_anchor = img_height/ratio
    num_col_anchor = img_width/ratio
    num_row_anchor = int(num_row_anchor)
    num_col_anchor = int(num_col_anchor)

    # register memory for anchors
    anchors = np.zeros(shape=(num_row_anchor, num_col_anchor, num_anchors, 4))

    # fill in data to empty arrays
    num_anchor_area = num_row_anchor * num_col_anchor
    for i in range(num_row_anchor):
        for j in range(num_col_anchor):
            x_center = j*ratio/img_width
            y_center = 1 - i*ratio/img_height
            pos_center = (x_center, y_center)
            anchors[i][j] = make_anchor_pyramid(anchor_scales, anchor_ratios, pos_center)


    return anchors

## Create an empty score-bbox reference map
#
# \pr{anchors, nested list, All anchors of an image.}
# \rt{score_bbox_map, nested list, An empty score-bbox reference map. Every}
# anchor's index corresponds to an empty score-bbox reference, [0.0, [] ].
def make_score_bbox_map(anchors):
    anchor_3d_shape = anchors.shape[:3]
    (iNum, jNum, kNum) = anchor_3d_shape
    score_bbox_map_shape = (iNum, jNum, kNum, 2)
    score_bbox_map = np.zeros(shape = score_bbox_map_shape, dtype='O')
    for i in range(iNum):
        for j in range(jNum):
            for k in range(kNum):
                score_bbox_map[i][j][k][0] = 0.0
                score_bbox_map[i][j][k][1] = []

    return score_bbox_map

## Update a score-bbox reference map given a bbox and all anchors
#
# \pr{score_bbox_map, nested_list, A created or previously updated score-bbox map.}
# \pr{bbox, list, A normalzied bounding box where elements are between 0 and 1.}
# \pr{anchors, nested list, Normalized anchors where elements' absolute}
# values are between 0 amd 1.
# \rt{score_bbox_map_result, nested list, The updated score-bbox map}
def update_score_bbox_map(score_bbox_map, bbox, anchors):
    score_bbox_map_result = score_bbox_map
    (iNum, jNum, kNum) = score_bbox_map.shape[:3]
    for i in range(iNum):
        for j in range(jNum):
            for k in range(kNum):
                anchor = anchors[i][j][k]
                overlap = intersection(bbox, anchor)
                if overlap == 0:
                    continue
                sum = union(bbox, anchor, overlap)
                score = overlap/sum
                if score > score_bbox_map[i][j][k][0]:
                    score_bbox_map_result[i][j][k][0] = score
                    score_bbox_map_result[i][j][k][1] = bbox
                # if score>0.7:
                #     #print('anchor',anchor)
                #     #print('bbox',bbox)
    return score_bbox_map_result

## Return an anchor label by IoU score and input limits
#
# If score < lim_lo, the anchor would be labeled as negative
# (False in classification).
# \n If score > lim_up, the anchor would be labeled as positive
# (True in classification)
# \n If lim_lo \f$\le\f$ score \f$\le\f$ lim_up, the anchor would be masked
# by nan and won't be used in training.
# \pr{score, float, The IoU score of an anchor.}
# \pr{lim_lo, float, The lower limit of the score range in which anchors should be masked.}
# \pr{lim_up, float, The upper limit of the score range in which anchors should be masked.}
# \rt{label, int, 0 }
# 1, or nan.
def label_anchor(score, lim_lo, lim_up):
    if score < lim_lo:
        #pdebug('Hello from labelling neg anchors')
        return 0
    elif score > lim_up:
        #pdebug('Hello from labelling pos anchors')
        return 1
    else:
        return np.nan

## Generate a label map that assigns every anchor a label.
#
# Label every anchor with a negative or positve or masked label.
# If score < lim_lo, the anchor would be labeled as negative
# (False in classification).
# \n If score > lim_up, the anchor would be labeled as positive
# (True in classification)
# \n If lim_lo \f$\le\f$ score \f$\le\f$ lim_up, the anchor would be masked
# by nan and won't be used for training.
# \pr{score-bbox-map, nested list, A fully updated score-bbox reference map.}
# \pr{lim_lo, float, The lower limit of the score range in which anchors should be masked.}
# \pr{lim_up, float, The upper limit of the score range in which anchors should be masked.}
# \rt{label_map, nested_list, A label map that every anchor has been assigned a label.}
def make_label_map(score_bbox_map, lim_lo, lim_up):
    shape = score_bbox_map.shape[:3]
    (iNum, jNum, kNum) = shape
    label_map = np.zeros(shape=shape, dtype=np.float32)
    label_map[:] = np.nan
    for i in range(iNum):
        for j in range(jNum):
            for k in range(kNum):
                label_map[i][j][k] = label_anchor(score_bbox_map[i][j][k][0], lim_lo, lim_up)
    return label_map

## Randomlly sample a label map for unbiased training
#
# \pr{label_map, nested list, A full label map generated by \c make_label_map.}
# \pr{posCut, int, Maximum number of positive anchors. If positive anchors are }
# less posCut, there would be nWant-posCut negative anchors used for training, if
# applicable.
# \pr{nWant, int, Minimum number of valid (positive or negative) anchors. If a }
# label map has less than nWant valid anchors, the function will continue but
# throw a warning.
# \rt{samples, nested list, Randomlly sampled label map. Masked anchors won't be
# used in backpropagation.}
def sample_label_map(label_map, posCut, nWant):
    (iNum, jNum, kNum) = label_map.shape
    samples = np.zeros(shape=(iNum, jNum, kNum))
    samples[:] = np.nan

    negNum = 0
    posNum = 0
    neg_index_list = []
    pos_index_list = []

    for i in range(iNum):
        for j in range(jNum):
            for k in range(kNum):
                if label_map[i][j][k] == 0:
                    negNum += 1
                    neg_index_list.append([i,j,k])
                elif label_map[i][j][k] == 1:
                    posNum += 1
                    pos_index_list.append([i,j,k])
                else:
                    pass

    totNum = negNum + posNum
    if totNum < nWant: # extreme case which may never happen
        neg_index_selected_list = neg_index_list
        pos_index_selected_list = pos_index_list
        pwarn(f'Extreme case detected in sampling label maps, negNum: {megNum}; posNum: {posNum}')
    elif posNum < posCut:
        # pdebug('Hello')
        neg_want = nWant-posNum
        neg_index_selected_list = random.sample(neg_index_list, neg_want)
        pos_index_selected_list = pos_index_list
    elif posNum < negNum:
        neg_want = posNum
        neg_index_selected_list = random.sample(neg_index_list, neg_want)
        pos_index_selected_list = pos_index_list
    else:
        pos_want = negNum
        neg_index_selected_list = neg_index_list
        pos_index_selected_list = random.sample(pos_index_list, pos_want)

    for i, j, k in pos_index_selected_list:
        samples[i][j][k] = 1

    for i, j, k in neg_index_selected_list:
        samples[i][j][k] = 0

    return samples

## Calculate delta given an anchor and a bounding box
#
# Return [tx, ty, tw, th] defined in the original paper.
# \pr{anchor, list, A list of normalized [xmin}
# , xmax, ymin, ymax].
# \pr{bbox, list, A list of normalized [xmin}
# , xmax, ymin, ymax].
# \rt{result, numpy array, Calculated [tx}
# , ty, tw, th].
def calc_delta(anchor, bbox):
    # anchor: xmin, xmax, ymin, ymax
    # bbox: xmin, xmax, ymin, ymax
    gx = (bbox[0] + bbox[1])/2
    gy = (bbox[2] + bbox[3])/2
    gw = bbox[1]-bbox[0]
    gh = bbox[3]-bbox[2]
    xa = (anchor[0] + anchor[1])/2
    ya = (anchor[2] + anchor[3])/2
    wa = anchor[1]-anchor[0]
    ha = anchor[3]-anchor[2]

    tx = (gx-xa)/wa
    ty = (gy-ya)/ha
    tw = math.log(gw/wa)
    th = math.log(gh/ha)

    result = np.array([tx, ty, tw, th])
    return result

## Generate a delta map that assigns every anchor that contains an object a delta.
#
# \pr{score_bbox_map, nested list, A score-bbox reference map.}
# \pr{lim_up, float, The upper limit of the score range in which an anchor will be masked.}
# Otherwise, the delta would be calculated and used for training RPN regression (delta output).
# \pr{anchors, nested_list, Every sublist is an anchor. Every anchor is a list of normalized [xmin}
# , xmax, ymin, ymax]. The anchor's parameters are required for calculating the delta.}
# \rt{delta_map, 3-D numpy array,The delta map that every anchor above lim_up is
# assigned a delta.}
def make_delta_map(score_bbox_map, lim_up, anchors):
    (iNum, jNum, kNum) = score_bbox_map.shape[:3]
    delta_shape = (iNum, jNum, kNum*4)
    delta_map = np.zeros(shape=delta_shape)
    delta_map[:] = np.nan
    for i in range(iNum):
        for j in range(jNum):
            for k in range(kNum):
                if score_bbox_map[i][j][k][0] > lim_up:
                    #pdebug('Hello! from making delta map')
                    anchor = anchors[i][j][k]
                    bbox = score_bbox_map[i][j][k][1]
                    delta = calc_delta(anchor, bbox)
                    delta_map[i][j][4*k] = delta[0]
                    delta_map[i][j][4*k+1] = delta[1]
                    delta_map[i][j][4*k+2] = delta[2]
                    delta_map[i][j][4*k+3] = delta[3]

    return delta_map

## Make a dictionary in which keys are img names and items are bboxes in the img.
# If an image is skipped (exists in img_dir but not in returned img_bbox_dict),
# there is no track in the image, which is possible due
# to the input probability distribution.
#
# \pr{img_dir, Path object, A directory that has training images.}
# \pr{bbox_file, Path object, A csv file that each line represents a bbox.}
# \rt{img_bbox_dict, dict, A dictionary in which keys are img names and items are bboxes in the img.}
def make_img_bbox_dict(img_dir, bbox_file):

    df = pd.read_csv(bbox_file, index_col=0)

    img_names = df['FileName'].unique()

    img_bbox_dict = {img : [] for img in img_names}
    def add_data(row):
        img_name = row.at['FileName']
        xmin = row.at['XMin']
        xmax = row.at['XMax']
        ymin = row.at['YMin']
        ymax = row.at['YMax']
        img_bbox_dict[img_name].append([xmin,xmax,ymin,ymax])

    df.apply(add_data, axis=1)

    return img_bbox_dict

## Given a delta, it tells an anchor what shape it should be.
#
# \pr{anchor, list, A normalized list of [xmin}
# , xmax, ymin, ymax].
# \pr{delta, list, A list of [tx}
# , ty, tw, th].
# \rt{result, list, A normalized bbox [xmin}
#, xmax, ymin, ymax].
def translate_delta(anchor, delta):

    tx, ty, tw, th = delta
    xa = (anchor[0]+anchor[1])/2
    ya = (anchor[2]+anchor[3])/2
    wa = anchor[1]-anchor[0]
    ha = anchor[3]-anchor[2]

    x = tx*wa+xa
    y = ty*ha+ya
    w = math.exp(tw)*wa
    h = math.exp(th)*ha

    xmin = x-w/2
    xmax = x+w/2
    ymin = y-h/2
    ymax = y+h/2

    result = [xmin, xmax, ymin, ymax]
    return result

## Propose bounding boxes and their scores. Only bboxes whose score > 0.5 will be proposed.
#
# \pr{anchors, nested list, Every sublist is an anchor. Every anchor is a list of normalized [xmin}
# , xmax, ymin, ymax].
# \pr{input_shape, list or tuple, The shape of input image}
# \pr{score_map, 3-D numpy array, The score output of RPN}
# \pr{delta_map, 3-D numpy array, The delta output of RPN}
# \rt{score_bbox_list, nested list, Every sublist contains a bbox and its score.}
# Explicitly, score_bbox_list[i] = [score, bbox].

def propose_score_bbox_list(anchors, score_map, delta_map):

    (iNum, jNum, kNum) = score_map.shape

    score_bbox_list = []
    threshold = 0.5
    for i, j, k in np.ndindex((score_map.shape)):
        score = score_map[i,j,k]
        if score > threshold:
            anchor = anchors[i][j][k]
            delta = delta_map[i,j,4*k:4*k+4]
            bbox = translate_delta(anchor, delta)
            score_bbox_list.append([score, bbox])

    return score_bbox_list
