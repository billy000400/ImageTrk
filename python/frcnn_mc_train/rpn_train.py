# A Faster R-CNN approach to Mu2e Tracking
# Compared to 00, the script should make essencial parameters portable and\
# use more Numpy and Pandas code to sandalize data I/O.
# Another goal is to make the base-NN modular for future improvement.
# Author: Billy Haoyang Li
# Email: li000400@umn.edu

### imports starts
import sys
from pathlib import Path
import pickle
import timeit
from datetime import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Layers import rpn
from Information import *
from Configuration import frcnn_config
from DataGenerator import DataGeneratorV2
from Loss import *
from Metric import *

### imports ends

def rpn_train(C, alternative=False):
    pstage("Start Training")

    # prepare data generator
    train_generator = DataGeneratorV2(C.train_img_inputs_npy, C.train_labels_npy, C.train_deltas_npy, batch_size=8)
    val_generator = DataGeneratorV2(C.validation_img_inputs_npy, C.validation_labels_npy, C.validation_deltas_npy, batch_size=8)

    # outputs
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
    weight_dir = C.data_dir.parent.joinpath('weights')
    C.weight_dir = weight_dir

    model_weights_file = weight_dir.joinpath(C.rpn_model_name+'.h5')
    record_file = data_dir.joinpath(C.rpn_record_name+'.csv')

    pinfo('I/O Path is configured')

    # build the model
    input_layer = Input(C.input_shape)
    base_net = C.base_net.get_base_net(input_layer, trainable=False)
    rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
    classifier = rpn_layer.classifier(base_net)
    regressor = rpn_layer.regression(base_net)
    model = Model(inputs=input_layer, outputs = [classifier,regressor])
    model.summary()

    if alternative:
        rpn_model_weights_file = C.weight_dir.joinpath(C.rpn_model_name+'.h5')
        detector_model_weights_file = C.weight_dir.joinpath('detector_mc_RCNN_dr=0.0.h5')
        pinfo(f"Loading weights from {str(rpn_model_weights_file)}")
        model.load_weights(rpn_model_weights_file, by_name=True)
        pinfo(f"Loading weights from {str(detector_model_weights_file)}")
        model.load_weights(detector_model_weights_file, by_name=True)

    # setup loss
    rpn_class_loss = define_rpn_class_loss(C.rpn_lambdas[0])
    rpn_regr_loss = define_rpn_regr_loss(C.rpn_lambdas[1])

    # setup optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # setup callbacks
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)


    # setup a callback to monitor number of positive anchors
    # if positive anchor number is less than limit, stop training
    class EarlyStoppingAtMinMetric(tf.keras.callbacks.Callback):
        def __init__(self, monitor, lo_limit):
            super(EarlyStoppingAtMinMetric, self).__init__()
            self.monitor = monitor
            self.lo_limit = lo_limit

        def on_train_batch_end(self, batch, logs=None):
            current = logs.get(self.monitor)
            if current < self.lo_limit:
                self.model.stop_training = True

    posNumCallback = EarlyStoppingAtMinMetric('rpn_out_class_positive_number', 30)

    earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    ModelCallback = tf.keras.callbacks.ModelCheckpoint(model_weights_file,\
                        monitor='val_loss', verbose=1,\
                        save_weights_only=True,\
                        save_best_only=True, mode='auto', save_freq='epoch')

    logdir="logs/fit/" + "rpn_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # compile the model
    model.compile(optimizer=adam, loss={'rpn_out_class' : rpn_class_loss,\
                                        'rpn_out_regress': rpn_regr_loss},\
                                    metrics={'rpn_out_class': [unmasked_binary_accuracy, positive_number],\
                                             'rpn_out_regress': unmasked_IoU})


    # initialize fit parameters
    model.fit(x=train_generator,
                validation_data=val_generator,\
                shuffle=True,\
                callbacks = [CsvCallback, ModelCallback, earlyStopCallback, tensorboard_callback],\
                epochs=200)


    model.save_weights(model_weights_file, overwrite=True)
    pinfo(f"Weights are saved to {str(model_weights_file)}")

    pickle_train_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pcheck_point('Finished Training')
    return C


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    pinfo('Parameters are set inside the script')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    lambdas = [1, 100]
    model_name = 'rpn_mc_00'
    record_name = 'rpn_mc_record_00'

    C.set_rpn_record(model_name, record_name)
    C.set_rpn_lambda(lambdas)
    C = rpn_train(C, alternative=True)
