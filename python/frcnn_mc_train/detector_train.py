# A Faster R-CNN approach to Mu2e Tracking
# The detector (Fast R-CNN) part for alternative (i) method in the original paper
# see https://arxiv.org/abs/1506.01497
# Author: Billy Haoyang Li
# Email: li000400@umn.edu

### imports starts
import sys
from pathlib import Path
import pickle
import timeit

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, TimeDistributed, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.metrics import CategoricalAccuracy

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config
from Layers import RoIPooling
from Loss import *
from Metric import *
### imports ends

def detector_train(C, alternative=False):
    pstage("Start Training")

    # load the oneHotEncoder
    oneHotEncoder = C.oneHotEncoder
    classNum = len(oneHotEncoder)

    # prepare the tensorflow.data.DataSet object
    inputs = np.load(C.img_inputs_npy)
    pinfo(f"Image Array Value Range: [{inputs.min()}, {inputs.max()}]")
    rois = np.load(C.rois)
    Y_labels = np.load(C.detector_train_Y_classifier)
    Y_bboxes = np.load(C.detector_train_Y_regressor)

    # outputs
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
    weights_dir = C.weight_dir

    rpn_model_weight_file = weights_dir.joinpath(C.rpn_model_name+'.h5')
    detector_model_weight_file = weights_dir.joinpath(C.detector_model_name+'.h5')
    record_file = data_dir.joinpath(C.detector_record_name+'.csv')

    pinfo('I/O Path is configured')

    # construct model
    img_input = Input(shape=C.input_shape)
    RoI_input = Input(shape=rois.shape[1:])
    x = C.base_net.get_base_net(img_input, trainable=True)
    x = RoIPooling(6,6)([x, RoI_input])
    x = TimeDistributed(Flatten(name='flatten'))(x)
    x = TimeDistributed(Dense(4096, activation='relu', name='fcl'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x1 = Dense(classNum)(x)
    output_classifier = Softmax(axis=2, name='detector_out_class')(x1)
    output_regressor = Dense(classNum*4, activation='linear', name='detector_out_regr')(x)

    model = Model(inputs=[img_input, RoI_input], outputs = [output_classifier, output_regressor])
    model.summary()

    # load weights trianed by RPN
    model.load_weights(rpn_model_weight_file, by_name=True)

    # setup loss functions
    detector_class_loss = define_detector_class_loss(C.detector_lambda[0])
    detector_regr_loss = define_detector_regr_loss(C.detector_lambda[1])

    # setup optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # setup callbacks
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)

    ca = CategoricalAccuracy()
    # compile the model
    model.compile(optimizer=adam, loss={'detector_out_class':detector_class_loss,\
                                        'detector_out_regr':detector_regr_loss},\
                                    metrics = {'detector_out_class':ca,\
                                                'detector_out_regr':unmasked_IoU})

    # initialize fit parameters
    model.fit(x=[inputs, rois], y=[Y_labels, Y_bboxes],\
                validation_split=0.25,\
                shuffle=True,\
                batch_size=8, epochs=100,\
                callbacks = [CsvCallback])

    model.save_weights(detector_model_weight_file, overwrite=True)

    pickle_train_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pcheck_point('Finished Training')


    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training Fast R-CNN detector')

    pinfo('Parameters are set inside the script')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    lambdas = [1, 100]
    model_name = 'detector_mc_RCNN_regr_linear'
    record_name = 'detector_mc_record_RCNN_regr_linear'

    C.set_detector_record(model_name, record_name)
    C.set_detector_lambda(lambdas)
    C = detector_train(C)
