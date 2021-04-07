
### imports starts
import sys
from pathlib import Path
import pickle
import cv2

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import frcnn_config
from DataGenerator import DataGeneratorV2
from Abstract import make_anchors, normalize_anchor, propose_score_bbox_list
from Layers import rpn
from Information import *
### import ends

def rpn_predict_RoI(C, nms=True):

    pstage('RPN is predicting Regions of Interest (RoIs) with NMS')
    #input_shape, ratio, anchor_scales, anchor_ratios
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)

    row_num, col_num = C.input_shape[:2]
    input_shape = [row_num, col_num]

    ### reconstruct model
    pinfo('Assembling model')
    # reconstruct model file
    cwd = Path.cwd()
    weights_dir = C.weight_dir
    model_weights = weights_dir.joinpath(C.rpn_model_name+'.h5')

    # load model
    input_layer = Input(shape=C.input_shape)
    x = C.base_net.get_base_net(input_layer)
    rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
    classifier = rpn_layer.classifier(x)
    regressor = rpn_layer.regression(x)

    model = Model(inputs=input_layer, outputs = [classifier, regressor])


    model.load_weights(model_weights, by_name=True)
    model.summary()


    ############################ Predicting by training data ###################################
    pinfo('Start predicting for training data')

    ### preparing input data
    train_generator = DataGeneratorV2(C.train_img_inputs_npy,\
                C.train_labels_npy, C.train_deltas_npy,\
                batch_size=1, shuffle=False)


    ### predicting by model
    pinfo('RPN is scoring anchors and proposing delta suggestions')
    outputs_raw = model.predict(x=train_generator)
    score_maps = outputs_raw[0]
    delta_maps = outputs_raw[1]

    ### filter out negative anchors and adjust positive anchors
    all_bbox = []
    img_idx = 0
    bbox_idx = 0
    dict_for_df={}

    bbox_df = pd.read_csv(C.train_bbox_reference_file, index_col=0)
    img_names = bbox_df['FileName'].unique().tolist()
    imgNum = len(img_names)

    if len(img_names) != len(score_maps):
        perr('Number of images is inconsistent with the number of outputs')
        sys.exit()

    for img_name, score_map, delta_map in zip(img_names, score_maps, delta_maps):

        sys.stdout.write(t_info(f'Proposing bounding boxes for image: {img_idx+1}/{imgNum}','\r'))
        if img_idx+1 == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # select all bbox whose objective score is >= 0.5 and put them in a list
        score_bbox_pairs = propose_score_bbox_list(anchors, score_map, delta_map)

        scores, bboxes_raw = [], []
        for score_bbox in score_bbox_pairs:
            scores.append(score_bbox[0])
            bboxes_raw.append(score_bbox[1])

        # trimming bboxes
        for i, bbox in enumerate(bboxes_raw):
            if bbox[0] < 0:
                bboxes_raw[i][0] = 0
            if bbox[1] > 1:
                bboxes_raw[i][1] = 1
            if bbox[2] < 0:
                bboxes_raw[i][2] = 0
            if bbox[3] > 1:
                bboxes_raw[i][3] = 1
            if bbox[1] < bbox[0]:
                pwarn(f"bbox {i} XMax is less than XMin")
            if bbox[3] < bbox[2]:
                pwarn(f"bbox {i} YMax is less than YMin")

        # NMS
        if nms == True:
            scores_tf = tf.constant(scores, dtype=tf.float32)
            bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
            bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)
            selected_indices, selected_scores =\
                non_max_suppression_with_scores(bboxes_raw_tf, scores_tf,\
                        max_output_size=100,\
                        iou_threshold=0.7, score_threshold=0.998,\
                        soft_nms_sigma=1000.0)

            selected_indices_list = selected_indices.numpy().tolist()
            bboxes = [ bboxes_raw[index] for index in selected_indices_list ]
            scores = selected_scores.numpy().tolist()
        else:
            bboxes = bboxes_raw
            scores = scores

        # save parameters to dict
        for score, bbox in zip(scores, bboxes):
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
    output_file = C.train_img_dir.parent.joinpath("mc_RoI_prediction_NMS_train.csv")
    output_df.to_csv(output_file)

    C.set_train_proposal(output_file)








    ############################ Predicting by validation data ###################################
    pinfo('Start predicting for validation data')







    ### preparing input data
    val_generator = DataGeneratorV2(C.validation_img_inputs_npy, C.validation_labels_npy, C.validation_deltas_npy, batch_size=1, shuffle=False)


    ### predicting by model
    pinfo('RPN is scoring anchors and proposing delta suggestions')
    outputs_raw = model.predict(x=val_generator)
    score_maps = outputs_raw[0]
    delta_maps = outputs_raw[1]

    ### filter out negative anchors and adjust positive anchors
    all_bbox = []
    img_idx = 0
    bbox_idx = 0
    dict_for_df={}

    bbox_df = pd.read_csv(C.validation_bbox_reference_file, index_col=0)
    img_names = bbox_df['FileName'].unique().tolist()
    imgNum = len(img_names)

    if len(img_names) != len(score_maps):
        perr('Number of images is inconsistent with the number of outputs')
        sys.exit()

    for img_name, score_map, delta_map in zip(img_names, score_maps, delta_maps):

        sys.stdout.write(t_info(f'Proposing bounding boxes for image: {img_idx+1}/{imgNum}','\r'))
        if img_idx+1 == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # select all bbox whose objective score is >= 0.5 and put them in a list
        score_bbox_pairs = propose_score_bbox_list(anchors, score_map, delta_map)

        scores, bboxes_raw = [], []
        for score_bbox in score_bbox_pairs:
            scores.append(score_bbox[0])
            bboxes_raw.append(score_bbox[1])

        # trimming bboxes
        for i, bbox in enumerate(bboxes_raw):
            if bbox[0] < 0:
                bboxes_raw[i][0] = 0
            if bbox[1] > 1:
                bboxes_raw[i][1] = 1
            if bbox[2] < 0:
                bboxes_raw[i][2] = 0
            if bbox[3] > 1:
                bboxes_raw[i][3] = 1
            if bbox[1] < bbox[0]:
                pwarn(f"bbox {i} XMax is less than XMin")
            if bbox[3] < bbox[2]:
                pwarn(f"bbox {i} YMax is less than YMin")

        # NMS
        if nms == True:
            scores_tf = tf.constant(scores, dtype=tf.float32)
            bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
            bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)
            selected_indices, selected_scores =\
                non_max_suppression_with_scores(bboxes_raw_tf, scores_tf,\
                        max_output_size=100,\
                        iou_threshold=0.7, score_threshold=0.998,\
                        soft_nms_sigma=1000.0)

            selected_indices_list = selected_indices.numpy().tolist()
            bboxes = [ bboxes_raw[index] for index in selected_indices_list ]
            scores = selected_scores.numpy().tolist()
        else:
            bboxes = bboxes_raw
            scores = scores

        # save parameters to dict
        for score, bbox in zip(scores, bboxes):
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
    output_file = C.train_img_dir.parent.joinpath("mc_RoI_prediction_NMS_validation.csv")
    output_df.to_csv(output_file)

    C.set_validation_proposal(output_file)

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pickle_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pcheck_point('Predicted bbox table')

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Proposing Region of Interest (RoI)')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    C = rpn_predict_RoI(C, nms=True)
