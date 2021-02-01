
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
from frcnn_config import Config
from frcnn_util import make_anchors, normalize_anchor, propose_bbox
from frcnn_rpn import rpn
from mu2e_output import *
### import ends

def rpn_test(C):

    pstage('Start testing RPN')
    anchors = make_anchors(C)

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
    inputs = np.load(cwd.parent.parent.joinpath('tmp').joinpath('inputs.npy'))

    ### predicting by model
    pinfo('RPN is scoring anchors and proposing delta suggestions')
    outputs_raw = model.predict(x=inputs, batch_size=1)
    score_maps = outputs_raw[0]
    # pdebug(score_maps.shape)
    delta_maps = outputs_raw[1]

    ### filter out negative anchors and adjust positive anchors
    all_bbox = []
    img_idx = 0
    bbox_idx = 0
    imgNum = inputs.shape[0]
    dict_for_df={}
    img_paths_raw = C.img_dir.glob('*')
    img_paths = [x for x in img_paths_raw if x.is_file()]

    for img_path, score_map, delta_map in zip(img_paths, score_maps, delta_maps):

        sys.stdout.write(t_info(f'Proposing bounding boxes for image: {img_idx+1}/{imgNum}','\r'))
        if img_idx+1 == imgNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # adjust all bbox whose objective score is >= 0.5 and put them in a list
        bbox_in_img_raw = propose_bbox(anchors, input_shape, score_map, delta_map)

        # Use soft NMS algorithm to resolve duplicate bbox
        bbox_in_img = bbox_in_img_raw

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

    C = rpn_test(C)

    pickle.dump(C, open(pickle_path, 'wb'))
