from pathlib import Path
import sys
import math
import random
import numpy as np
import pandas as pd
from mu2e_output import *

# binning objects
def binning_objects(objects, features, bin_array):

    assert len(objects)==len(features), \
        t_error('The lengths of object and feature are not equal')

    objs = objects
    ftrs = features
    bins = sorted(bin_array)

    try:
        objs = objs.to_list()
        ftrs = ftrs.to_list()
        bins = bins.to_list()
    except:
        pass

    obj_ftr_raw  = dict(zip(objs,ftrs))
    obj_ftr = sorted( obj_ftr_raw.items(), key=lambda item: item[1])

    result = []
    obj_ftr_idx = 0
    max_idx = len(obj_ftr)-1
    for idx in range(len(bins)):
        bin_result = []
        if obj_ftr_idx == max_idx:
            result.append(bin_result)
            continue

        if idx == 0:
            bin_left = float("-inf")
            bin_right = bins[0]
        else:
            bin_left = bins[idx-1]
            bin_right = bins[idx]

        while obj_ftr_idx <= max_idx:
            if (obj_ftr[obj_ftr_idx][1] >= bin_left) and (obj_ftr[obj_ftr_idx][1] < bin_right):
                bin_result.append(obj_ftr[obj_ftr_idx][0])
                obj_ftr_idx += 1
            else:
                break

        result.append(bin_result)

    return result

# anchor related
def normalize_anchor(anchor, input_shape):
    """
    Arguments:
        anchor: unnormalized list [xmin, xmax, ymin, ymax]
        input_shape: list [row number, col number]
    Return:
        normalized list [xmin, xmax, ymin, ymax]
    """
    width = input_shape[1]
    height = input_shape[0]
    xmin = anchor[0]/width
    xmax = anchor[1]/width
    ymin = anchor[2]/height
    ymax = anchor[3]/height

    return [xmin, xmax, ymin, ymax]

def make_anchor_pyramid(anchor_scales, anchor_ratios, input_shape, pos_center):
    """
    Arguments:
        anchor_scales: unnormalized anchor scales(list)
        anchor_ratios: Normalzied anchor_ratios(list of lists),
                        every list is like [a, b], in which a*b=1
        input_shape: unnormalized shape of input images
        pos_center: Unormalzied position of the pyramid's center
    Return:
        a list of anchors at this position
        each anchor is a list of normalized [xmin, xmax, ymin, ymax]
    """
    img_height = input_shape[0]
    img_width = input_shape[1]
    x = pos_center[0]
    y = pos_center[1]
    num_ratios = len(anchor_ratios)
    num_anchors = len(anchor_scales) * num_ratios

    pyramid = np.zeros(shape=(num_anchors,4))
    for i1, scale in enumerate(anchor_scales):
        for i2, xy_set in enumerate(anchor_ratios):
            for j, side in enumerate(xy_set):
                if j == 0: # width
                    width = scale*side/img_width
                    xmin = x-width/2
                    xmax = x+width/2
                    pyramid[i1*num_ratios+i2][0]= xmin
                    pyramid[i1*num_ratios+i2][1]= xmax
                else: # height
                    height = scale*side/img_height
                    ymin = y-height/2
                    ymax = y+height/2
                    pyramid[i1*num_ratios+i2][2] = ymin
                    pyramid[i1*num_ratios+i2][3] = ymax

    return pyramid

def make_anchors(input_shape, ratio, anchor_scales, anchor_ratios):
    """
    Arguments:
        Configuration object
    Return:
        3d list, first 2 indices represents the column and row number of the anchor pyramid
        the last index indicates a specific anchor in the anchor pyramid
        every anchor is a list of unnormalized [xmin, xmax, ymin, ymax]
    """

    # Check if the baseNet is set
    # This is necessary for determining how many pixels
    # an anchor represents


    # check if anchor scales and ratios are set

    # load parameters from configuration

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
            anchors[i][j] = make_anchor_pyramid(anchor_scales, anchor_ratios, input_shape, pos_center)


    return anchors

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
    return score_bbox_map_result

def label_anchor(score, lim_lo, lim_up):
    if score < lim_lo:
        #pdebug('Hello from labelling neg anchors')
        return 0
    elif score > lim_up:
        #pdebug('Hello from labelling pos anchors')
        return 1
    else:
        return np.nan

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

    return np.array([tx, ty, tw, th])

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

### bbox related
def make_bbox_dict(C):

    pinfo('Making the image-bbox dictionary')
    img_dir = C.img_dir
    bbox_file = C.bbox_file

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

def translate_delta(anchor, delta):
    """
    Arguments:
        anchor: a normalized anchor, a list of [xmin, xmax, ymin, ymax]
        delta: a delta list [tx, ty, tw, th]
    Return:
        a normalized bbox: [xmin, xmax, ymin, ymax]
    """
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

    return [xmin, xmax, ymin, ymax]

def propose_score_bbox_list(anchors, input_shape, score_map, delta_map):
    """
    Arguments:
        anchors: 3d nested list, see "make_anchors()"
        score_map: output numpy array of RPN
        delta_map: output numpy array of RPN
    Return:
        a list of [score, bbox]
    """
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
