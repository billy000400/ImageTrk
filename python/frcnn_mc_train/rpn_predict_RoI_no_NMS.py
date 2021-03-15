
### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Config import frcnn_config as Config
from Abstract import make_anchors, normalize_anchor, propose_score_bbox_list
from frcnn_rpn import rpn
from mu2e_output import *
### import ends

def rpn_test(C):

    pstage('RPN is predicting Regions of Interest (RoIs) without NMS')
    #input_shape, ratio, anchor_scales, anchor_ratios
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)

    row_num, col_num = C.input_shape[:2]
    input_shape = [row_num, col_num]

    ### reconstruct model
    pinfo('Reconstructing model')
    # reconstruct model file
    cwd = Path.cwd()
    weights_dir = cwd.parent.parent.joinpath('weights')
    model_weights = weights_dir.joinpath(C.model_name+'.h5')
    pdebug(model_weights)

    # load model
    rpn = C.set_rpn()

    input_layer = Input(shape=C.input_shape)
    x = C.base_net.nn(input_layer)
    classifier = rpn.classifier(x)
    regressor = rpn.regression(x)
    model = Model(inputs=input_layer, outputs = [classifier,regressor])


    model.load_weights(model_weights, by_name=True)

    ### preparing input data
    pinfo('Loading the original input array')
    inputs = np.load(C.inputs_npy)

    ### predicting by model
    pinfo('RPN is scoring anchors and proposing delta suggestions')
    outputs_raw = model.predict(x=inputs, batch_size=1)
    score_maps = outputs_raw[0]
    delta_maps = outputs_raw[1]

    ### filter out negative anchors and adjust positive anchors
    all_bbox = []
    img_idx = 0
    bbox_idx = 0
    imgNum = inputs.shape[0]
    dict_for_df={}

    bbox_df = pd.read_csv(C.bbox_file, index_col=0)
    img_names = bbox_df['FileName'].unique().tolist()

    if len(img_names) != len(score_maps):
        perr('Number of images is inconsistent with the number of outputs')
        sys.exit()

    for img_name, score_map, delta_map in zip(img_names, score_maps, delta_maps):

        sys.stdout.write(t_info(f'Proposing bounding boxes for image: {img_idx+1}/{imgNum}','\r'))
        if img_idx+1 == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # select all bbox whose objective score is >= 0.5 and put them in a list
        score_bbox_pairs = propose_score_bbox_list(anchors, input_shape, score_map, delta_map)

        scores, bbox_raws = [], []
        for score_bbox in score_bbox_pairs:
            scores.append(score_bbox[0])
            bbox_raws.append(score_bbox[1])

        # trimming bboxes
        for i, bbox in enumerate(bbox_raws):
            if bbox[0] < 0:
                bbox_raws[i][0] = 0
            if bbox[1] > 1:
                bbox_raws[i][1] = 1
            if bbox[2] < 0:
                bbox_raws[i][2] = 0
            if bbox[3] > 1:
                bbox_raws[i][3] = 1
            if bbox[1] < bbox[0]:
                pwarn(f"bbox {i} XMax is less than XMin")
            if bbox[3] < bbox[2]:
                pwarn(f"bbox {i} YMax is less than YMin")

        # save parameters to dict
        for score, bbox in zip(scores, bbox_raws):
            dict_for_df[bbox_idx] = {'FileName': str(img_name),\
                                'XMin':bbox[0],\
                                'XMax':bbox[1],\
                                'YMin':bbox[2],\
                                'YMax':bbox[3],\
                                'Score':score}
            bbox_idx += 1

        img_idx += 1

    # save proposed bboxes to local
    output_df = pd.DataFrame.from_dict(dict_for_df, "index")
    output_file = C.img_dir.parent.joinpath("mc_RoI_prediction_no_NMS.csv")
    output_df.to_csv(output_file)

    C.set_prediction(output_file)

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.test.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pcheck_point('Predicted bbox table')

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Testing')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    C = rpn_test(C)

    pickle.dump(C, open(pickle_path, 'wb'))
