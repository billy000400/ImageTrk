import sys
from pathlib import Path
import shutil
import timeit
import pickle
from copy import deepcopy
import math
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import wcnn_config
from Architectures import Img2VecV4
from DataGenerator import DataGeneratorV2
from Loss import *
from Metric import *

def train(C):

    # prepare data generator
    train_generator = DataGeneratorV2(C.X_train_dir, C.Y1_train_dir, C.Y2_train_dir, batch_size=1)
    val_generator = DataGeneratorV2(C.X_val_dir, C.Y1_val_dir, C.Y2_val_dir, batch_size=1)

    # outputs
    data_dir = C.sub_data_dir
    weight_dir = C.data_dir.parent.joinpath('weights')
    weight_dir.mkdir(parents=True, exist_ok=True)
    C.weight_dir = weight_dir

    model_weights_file = weight_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    pinfo('I/O Path is configured')

    # build the model
    model = Img2VecV4(input_shape=(256, 256, 1)).get_model()
    model.summary()

    classifier_loss = define_rpn_class_loss(1, weight=C.weights)
    # print(C.weights)
    # sys.exit()
    regressor_loss = define_rpn_regr_loss(10)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # setup callbacks
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)

    earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    ModelCallback = tf.keras.callbacks.ModelCheckpoint(str(model_weights_file),\
                        monitor='val_loss', verbose=1,\
                        save_weights_only=True,\
                        save_best_only=True, mode='min', save_freq='epoch')

    logdir="logs/fit/" + "wcnn_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # compile the model
    model.compile(optimizer=adam, loss={'classifier': classifier_loss,\
                                        'regressor': regressor_loss},
                                    metrics={'classifier': [unmasked_precision, unmasked_recall, positive_number],\
                                            'regressor': unmasked_IoU1D})

    # initialize fit parameters
    model.fit(x=train_generator,
                validation_data=val_generator,\
                shuffle=True,\
                callbacks = [CsvCallback, ModelCallback, tensorboard_callback],\
                epochs=1000)

    model.evaluate(x=train_generator)

    # model.save(model_weights_file, overwrite=True)

    pinfo(f"Weights are saved to {str(model_weights_file)}")
    pcheck_point('Finished Training')
    return C

if __name__ == "__main__":
    pbanner()
    psystem('Window-based Convolutional Neural Network')
    pmode('Generating validating data')
    pinfo('Input DType for testing: StrawHit')

    # load configuration
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('wcnn.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    model_name = 'wcnn_01_weight_1_10_res64'
    record_name = 'wcnn_record_01_weight_1_10_res64'
    C.set_outputs(model_name, record_name)

    C = train(C)
    pickle.dump(C, open(pickle_path, 'wb'))
