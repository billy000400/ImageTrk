### Not working!!!
### Alternative trianing of Faster R-CNN,
# which means we are going to train RPN first, then detector, then RPN, then detector...

import sys
from pathlib import Path
import pickle
import random
from copy import copy

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.image import non_max_suppression_with_scores
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import *
from Information import *
from Configuration import frcnn_config
from Layers import rpn, RoIPooling
from Loss import *
from Metric import *
from Information import *


def rpn_to_roi(C, score_maps, delta_maps):
    ### pass output through nms
    # make anchors
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)

    # initialize parameters
    bbox_idx = 0
    dict_for_df={}


    # prepare reference and prediction dataframes
    df_r = pd.read_csv(C.bbox_reference_file, index_col=0)
    imgNames = df_r['FileName'].unique().tolist()

    for img_name, score_map, delta_map in zip(imgNames, score_maps, delta_maps):
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

        scores_tf = tf.constant(scores, dtype=tf.float32)
        bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
        bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)
        selected_indices, selected_scores =\
            non_max_suppression_with_scores(bboxes_raw_tf, scores_tf,\
                max_output_size=100,\
                iou_threshold=0.7, score_threshold=0.9,\
                soft_nms_sigma=0.0)

        selected_indices_list = selected_indices.numpy().tolist()
        bboxes = [ bboxes_raw[index] for index in selected_indices_list ]

        for score, bbox in zip(scores, bboxes):
            dict_for_df[bbox_idx] = {'FileName': str(img_name),\
                                'XMin':bbox[0],\
                                'XMax':bbox[1],\
                                'YMin':bbox[2],\
                                'YMax':bbox[3],\
                                'Score':score}
            bbox_idx += 1


    df_p = pd.DataFrame.from_dict(dict_for_df, "index")


    ### make training data for detector
    # register memory for input and output data
    oneHotEncoder = C.oneHotEncoder

    inputs = np.load(C.img_inputs_npy)
    imgNum = inputs.shape[0]
    rois = np.zeros(shape=(imgNum, C.roiNum, 4))
    rois[:] = np.nan
    outputs_classifier = np.zeros(shape=(imgNum, C.roiNum, len(oneHotEncoder)), dtype=np.float32)
    outputs_classifier[:] = np.nan
    outputs_regressor = np.zeros(shape=(imgNum, C.roiNum, len(oneHotEncoder)*4), dtype=np.float32)
    outputs_regressor[:] = np.nan



    # calculate how many negative examples we want
    negThreshold = np.int(C.roiNum*C.negativeRate)

    for img_idx, img in enumerate(imgNames):

        sys.stdout.write(t_info(f"Parsing image: {img_idx+1}/{len(imgNames)}", '\r'))
        if img_idx+1 == len(imgNames):
            sys.stdout.write('\n')
        sys.stdout.flush()

        df_r_slice = df_r[df_r['FileName']==img]
        df_p_slice = df_p[df_p['FileName']==img]

        bbox_pdgId_pairs =\
         [   [[r['XMin'], r['XMax'], r['YMin'], r['YMax']],\
                        r['ClassName']] \
                 for i, r in df_r_slice.iterrows()]
        proposals = [ [r['XMin'], r['XMax'], r['YMin'], r['YMax']] for i, r in df_p_slice.iterrows()]

        pos_tuples = []
        neg_tuples = []

        # iterate over proposed bboxes to sort them into positives and negatives
        for proposal in proposals:
            iou_highest=0
            label = None
            ref_bbox = None
            for bbox, pdgId in bbox_pdgId_pairs:
                iou_tmp = iou(bbox, proposal)
                if iou_tmp > iou_highest:
                    iou_highest = iou_tmp
                    label = pdgId
                    ref_bbox = bbox
            if iou_highest > 0.5:
                pos_tuples.append((proposal, label, ref_bbox))
            elif iou_highest > 0.1:
                neg_tuples.append((proposal, 'bg', ref_bbox))

        # calculate the number of positive example and negative example
        posNum = len(pos_tuples)
        negNum = len(neg_tuples)
        totNum = posNum + negNum

        roiNum = C.roiNum

        if totNum < roiNum:
            tuples_combined = pos_tuples+neg_tuples # The original/whole sample
            sampleNum = len(tuples_combined)

            tuples_selected = copy(tuples_combined)
            totNum = len(tuples_selected)
            roiNeedNum = roiNum-totNum

            while roiNeedNum != 0:

                if sampleNum < roiNeedNum:
                    tuples_selected += tuples_combined
                    totNum = len(tuples_selected)
                    roiNeedNum = roiNum - totNum
                else:
                    tuples_selected += random.sample(tuples_combined, roiNeedNum)
                    totNum = len(tuples_selected)
                    roiNeedNum = roiNum - totNum

            assert len(tuples_selected)==roiNum, pdebug(len(tuples_selected))

        else:

            if negNum < negThreshold:
                negWant = negNum
                posWant = roiNum - negWant
            elif (negThreshold + posNum) >= roiNum:
                negWant = negThreshold
                posWant = roiNum - negThreshold
            else:
                posWant = posNum
                negWant = roiNum - posWant

            # randomly select RoIs for training
            pos_selected = random.sample(pos_tuples, posWant)
            neg_selected = random.sample(neg_tuples, negWant)

            # combine negative examples and positive examples and shuffle
            tuples_selected = pos_selected + neg_selected

        random.shuffle(tuples_selected)

        # copy the result to the registered memory
        for i, tuple in enumerate(tuples_selected):
            proposal, label, ref_bbox = tuple
            # proposal t = (x, y, w, h) as indicated in the original paper
            # (x,y) is the left upper corner
            t = [ proposal[0], proposal[3],\
                        (proposal[1]-proposal[0]), (proposal[3]-proposal[2]) ]
            rois[img_idx][i] = np.array(t, dtype=np.float32)

            oneHotVector = oneHotEncoder[label]
            outputs_classifier[img_idx][i] = oneHotVector

            # refernece bbox v = (x,y,w,h) as indicated in the original paper
            # (x,y) is the left upper corner
            v = [ ref_bbox[0], ref_bbox[3],\
                        (ref_bbox[1]-ref_bbox[0]), (ref_bbox[3]-ref_bbox[2]) ]
            record_start = np.where(oneHotVector==1)[0][0] *4
            record_end = record_start + 4
            outputs_regressor[img_idx][i][record_start:record_end] = v

    return rois, outputs_classifier, outputs_regressor

def frcnn_train_alternative(C):
    pstage('Start Training')

    # prepare oneHotEncoder
    df_r = pd.read_csv(C.bbox_reference_file, index_col=0)
    categories = df_r['ClassName'].unique().tolist()
    oneHotEncoder = {}
    # The first entry indicates if it is a negative example (background)
    oneHotEncoder['bg'] = np.identity(len(categories)+1)[0]
    for i, pdgId in enumerate(categories):
        oneHotEncoder[pdgId] = np.identity(len(categories)+1)[i+1]
    C.set_oneHotEncoder(oneHotEncoder)


    # prepare training dataset
    inputs = np.load(C.img_inputs_npy)
    label_maps = np.load(C.labels_npy)
    delta_maps = np.load(C.deltas_npy)

    # outputs
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
    weight_dir = C.data_dir.parent.joinpath('weights')
    C.weight_dir = weight_dir

    model_weights_file = weight_dir.joinpath(C.frcnn_model_name+'.h5')
    rpn_record_file = data_dir.joinpath(C.frcnn_record_name+'_rpn.csv')
    detector_record_file = data_dir.joinpath(C.frcnn_record_name+'_detector.csv')
    pinfo('I/O Path is configured')

    # build models
    img_input = Input(C.input_shape)
    base_net = C.base_net.get_base_net(img_input)
    rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
    rpn_classifier = rpn_layer.classifier(base_net)
    rpn_regressor = rpn_layer.regression(base_net)
    model_rpn = Model(inputs=img_input, outputs = [rpn_classifier,rpn_regressor])

    RoI_input = Input(shape=(C.roiNum,4))
    x = RoIPooling(6,6)([base_net, RoI_input])
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Reshape((C.roiNum,6*6*4096))(x)
    x1 = Dense(2)(x)
    output_classifier = Softmax(axis=2, name='detector_out_class')(x1)
    output_regressor = Dense(2*4, activation='linear', name='detector_out_regr')(x)
    model_detector = Model(inputs=[img_input, RoI_input], outputs = [output_classifier, output_regressor])

    model_all = Model(inputs=[img_input, RoI_input], outputs=[rpn_classifier,rpn_regressor,output_classifier, output_regressor])

    # setup loss
    rpn_regr_loss = define_rpn_regr_loss(C.rpn_lambdas[1])
    rpn_class_loss = define_rpn_class_loss(C.rpn_lambdas[0])
    detector_class_loss = define_detector_class_loss(C.detector_lambda[0])
    detector_regr_loss = define_detector_regr_loss(C.detector_lambda[1])

    # setup metric
    ca = CategoricalAccuracy()

    # setup optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # compile models
    model_rpn.compile(optimizer=adam, loss={'rpn_out_class' : rpn_class_loss,\
                                        'rpn_out_regress': rpn_regr_loss},\
                                    metrics={'rpn_out_class': [unmasked_binary_accuracy, positive_number],\
                                             'rpn_out_regress': unmasked_IoU})

    model_detector.compile(optimizer=adam, loss={'detector_out_class':detector_class_loss,\
                                        'detector_out_regr':detector_regr_loss},\
                                    metrics = {'detector_out_class':ca,\
                                                'detector_out_regr':unmasked_IoU})

    model_all.compile(optimizer=adam, loss={'rpn_out_class' : rpn_class_loss,\
                                        'rpn_out_regress': rpn_regr_loss,\
                                        'detector_out_class':detector_class_loss,\
                                        'detector_out_regr':detector_regr_loss})
    model_all.summary()
    # setup record
    RpnCsvCallback = tf.keras.callbacks.CSVLogger(str(rpn_record_file), separator=",", append=True)
    DetectorCsvCallback = tf.keras.callbacks.CSVLogger(str(detector_record_file), separator=",", append=True)

    time_of_iterations = 2

    for i in range(time_of_iterations):
        pinfo(f"Alternative training: iteration {i+1}/{time_of_iterations}")

        pinfo('RPN training')
        model_rpn.fit(x=inputs, y=[label_maps, delta_maps],\
                    validation_split=0.25,\
                    shuffle=True,\
                    batch_size=8, epochs=10,\
                    callbacks = [RpnCsvCallback])
        pinfo('RPN is scoring anchors and proposing delta suggestions')
        outputs_raw = model_rpn.predict(x=inputs, batch_size=1)

        pinfo('Proposing RoIs')
        score_maps = outputs_raw[0]
        delta_maps = outputs_raw[1]
        rois, Y_labels, Y_bboxes = rpn_to_roi(C, score_maps, delta_maps)

        pinfo('Detector training')
        model_detector.fit(x=[inputs, rois], y=[Y_labels, Y_bboxes],\
                    validation_split=0.25,\
                    shuffle=True,\
                    batch_size=8, epochs=1,\
                    callbacks = [DetectorCsvCallback])

    model_all.save_weights(model_weights_file, overwrite=True)
    pickle_train_path = Path.cwd().parent.joinpath('frcnn_mc_test/frcnn.test.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pcheck_point('Finished Training')

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Alternative Training')

    pinfo('Parameters are set inside the script')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path, 'rb'))

    # initialize parameters
    roiNum = 100
    negativeRate = 0.75

    rpn_lambdas = [1,100]
    detector_lambdas = [1,100]
    model_name = 'frcnn_mc_00'
    record_name = 'frcnn_mc_record_00'

    C.set_roi_parameters(roiNum, negativeRate)

    C.set_rpn_lambda(rpn_lambdas)
    C.set_detector_lambda(detector_lambdas)
    C.set_frcnn_record(model_name, record_name)
    C = frcnn_train_alternative(C)
