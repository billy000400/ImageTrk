import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, TimeDistributed, Reshape, Softmax

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import make_anchors
from Information import *
from Configuration import frcnn_config
from Layers import *

class track_detection:
    def __init__(self, C):
        self.config = C
        self.rpn_model = None
        self.detector_model = None
        self.__build_models(self.config)

    def __build_models(self, C):
        self.__build_rpn_model(C)
        self.__build_detector_model(C)
        return

    def __build_rpn_model(self, C):
        img_input = Input(shape=C.input_shape, name='x1')
        x = C.base_net.get_base_net(img_input, trainable=False)
        rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
        rpn_cls = rpn_layer.classifier(x)
        rpn_rgr = rpn_layer.regression(x)

        model_rpn = Model(inputs=img_input, outputs=[x, rpn_cls, rpn_rgr])

        model_rpn.load_weights(\
                str(cwd.joinpath(C.rpn_model_name+'.h5')), by_name=True)
        model_rpn.load_weights(\
                str(cwd.joinpath(C.detector_model_name+'.h5')), by_name=True)

        self.rpn_model = model_rpn

    def __build_detector_model(self, C):
        detector_fetch = Input(shape=(32, 32, 512), name='base_to_detector')
        RoI_input = Input(shape=(None, 4), name='roi')
        x = RoIPooling(6,6)([detector_fetch, RoI_input])
        x = TimeDistributed(Flatten(name='flatten'))(x)
        x = TimeDistributed(Dense(4096, activation='relu', name='fcl', trainable=False))(x)
        x = TimeDistributed(Dense(4096, activation='relu', name='fc2', trainable=False))(x)
        classNum = 2
        x1 = TimeDistributed(Dense(classNum, name='fc3', trainable=False))(x)
        x2 = TimeDistributed(Dense(classNum*4, activation='relu', name='fc4', trainable=False))(x)
        output_classifier = TimeDistributed(Softmax(axis=-1), name='detector_out_class')(x1)
        output_regressor = TimeDistributed(Dense(classNum*4, activation='linear', trainable=False), name='detector_out_regr')(x2)

        model_detector = Model(inputs=[detector_fetch, RoI_input], outputs = [output_classifier, output_regressor])
        model_detector.load_weights(\
                str(cwd.joinpath(C.detector_model_name+'.h5')), by_name=True)

        self.detector_model = model_detector

    def __make_image(self):
        return

    def detect(self, hits):
        data = self.__make_image(hits)
        ftr_maps, rpn_score_maps, rpn_delta_maps=\
                            self.rpn_model.predict_on_batch(x)
        rpn_scores = np.squeeze(rpn_score_maps, axis=0)
        rpn_deltas = np.squeeze(rpn_delta_maps, axis=0)

        rois, rpn_scores = rpn_to_roi(rpn_scores, rpn_deltas)
        rpn_proposals = xywh_to_bbox(rois).numpy()
        rpn_scores = rpn_scores.numpy()

        rois = np.expand_dims(rois, axis=0)
        detector_label_maps, detector_delta_maps =\
                        self.detector_model.predict_on_batch([ftr_maps, rois])

        detector_labels = np.squeeze(detector_label_maps, axis=0)
        detector_deltas = np.squeeze(detector_delta_maps, axis=0)
        detector_deltas = np.reshape(detector_deltas, (detector_deltas.shape[0],-1, 4))

        maxIndexes = np.argmax(detector_labels, axis=1)
        bg_mask = maxIndexes!=0 # only want bg_score < 0.5 proposals
        frcnn_proposals = rpn_proposals[bg_mask]

        ### calculate non-bg score
        selected_labels = detector_labels[bg_mask]
        frcnn_scores = np.amax(selected_labels, axis=1)

        deltas_raws = detector_deltas[bg_mask]
        oj_indexes = np.argmax(selected_labels, axis=1)
        deltas = []
        for j,oj_index in enumerate(oj_indexes):
            deltas.append(deltas_raws[j,oj_index,:])
        deltas = np.array(deltas, dtype=np.float32)

        return frcnn_proposals
