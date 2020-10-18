
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
from tensorflow.image import non_max_suppression_with_scores

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_config import Config
from frcnn_util import make_anchors, normalize_anchor, propose_score_bbox_list
from frcnn_rpn import rpn
from mu2e_output import *
### import ends


def predict(C, Nt=0, Ot=0, Sigma=0):

    pstage('Start predicting bounding boxes')

    ### unpacking parameters from configuration
    img_dir = C.test_img_dir
    #img_dir = Path.cwd().parent.parent.joinpath('data').joinpath('imgs_train')

    ### prepare an anchor map and the input shape
    input_shape = C.input_shape[:2]
    anchors = make_anchors(input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)

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
    pinfo('Loading testing images')
    # get the image file list
    inputs = []
    img_paths_raw = img_dir.glob('*')
    img_paths = [x for x in img_paths_raw if x.is_file()]
    imgNum = len(img_paths)
    for idx, img_path in enumerate(img_paths):
        sys.stdout.write(t_info(f'Loading testing image: {idx+1}/{imgNum}', '\r'))
        if (idx+1) == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        img_path_str = str(img_path)
        input = cv2.imread(img_path_str)
        if input.any() == None:
            perr(f'{img_path_str} is invalid')
        inputs.append(input)
    inputs = np.asarray(inputs)
    #inputs = np.load(cwd.parent.parent.joinpath('tmp').joinpath('inputs.npy'))

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
    for img_path, score_map, delta_map in zip(img_paths, score_maps, delta_maps):
        sys.stdout.write(t_info(f'Proposing bounding boxes for image: {img_idx+1}/{imgNum}','\r'))
        if img_idx+1 == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # adjust all bbox whose objective score is >= 0.5 and put them in a list
        score_bbox_raw = propose_score_bbox_list(anchors, input_shape, score_map, delta_map)

        # Use soft NMS algorithm to resolve duplicate bbox
        score = [ score for score, bbox in score_bbox_raw ]
        score = tf.constant(score, dtype=tf.float32)

        bbox_raw = [ bbox for score, bbox in score_bbox_raw ]

        # trimming bboxes
        for i, bbox in enumerate(bbox_raw):
            if bbox[0] < 0:
                bbox_raw[i][0] = 0
            if bbox[1] > 1:
                bbox_raw[i][1] = 1
            if bbox[2] < 0:
                bbox_raw[i][2] = 0
            if bbox[3] > 1:
                bbox_raw[i][3] = 1
            if bbox[1] < bbox[0]:
                pwarn(f"bbox {i} XMax is less than XMin")
            if bbox[3] < bbox[2]:
                pwarn(f"bbox {i} YMax is less than YMin")

        bbox_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bbox_raw ]
        bbox_raw_tf = tf.constant(bbox_raw_tf, dtype=tf.float32)

        if Nt == 0:
            selected_indices, selected_score =\
            non_max_suppression_with_scores(bbox_raw_tf, score,\
                        max_output_size=20,\
                        iou_threshold=1.0, score_threshold=0.9,\
                        soft_nms_sigma=0.0)
        else:
            selected_indices, selected_score =\
            non_max_suppression_with_scores(bbox_raw_tf, score,\
                        max_output_size=20,\
                        iou_threshold=Nt, score_threshold=Ot,\
                        soft_nms_sigma=Sigma)

        selected_indices_list = selected_indices.numpy().tolist()
        bbox_in_img = [ bbox_raw[index] for index in selected_indices_list ]

        selected_score = selected_score.numpy().tolist()

        # save parameters to dict
        if len(bbox_in_img) == 0:
            pwarn(f"No proposal for {img_path.name}")
            continue

        for i, bbox in enumerate(bbox_in_img):
            img_name = img_path.name
            dict_for_df[bbox_idx] = {'FileName': str(img_name),\
                                'XMin':bbox[0],\
                                'XMax':bbox[1],\
                                'YMin':bbox[2],\
                                'YMax':bbox[3]}
            bbox_idx += 1

        img_idx += 1

    # save proposed bboxes to local
    output_df = pd.DataFrame.from_dict(dict_for_df, "index")
    output_file = C.test_bbox_table_reference.parent.joinpath("bbox_proposal_test_prediction.csv")
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
    pickle_path = cwd.joinpath('frcnn.test.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    C = predict(C)
