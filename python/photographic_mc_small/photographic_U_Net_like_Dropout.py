# This script's purpose is to train a preliminary CNN for tracking by Keras
# Author: Billy Li
# Email: li000400@umn.edu

# import starts
import sys
from pathlib import Path
import csv
import random
import pickle

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, initializers, regularizers
from tensorflow.keras.layers import(
    Input,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPool2D,Dropout,
    Flatten,
    TimeDistributed,
    Embedding,
    Reshape,
    Softmax
)
from tensorflow.keras.optimizers import Adam


util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Config import extractor_config as Config
from Architecture import U_Net_Like
from mu2e_output import *
from Loss import *
from Metric import *
### import ends

def photographic_train(C):
    pstage("Start Training")

    ### load inputs
    pinfo('Loading processed arrays')
    X = np.load(C.X_npy)
    Y = np.load(C.Y_npy)

    pinfo('Standarlizing input arrays')
    mean = X.mean()
    std = X.std()
    std_inv = 1/std
    X = (X-mean)*std_inv

    pinfo('Calculating class weights by median frequency')
    blankNum = np.count_nonzero( (Y==np.array([1,0,0])).all(axis=3) )
    bgNum = np.count_nonzero( (Y==np.array([0,1,0])).all(axis=3) )
    majorNum = np.count_nonzero( (Y==np.array([0,0,1])).all(axis=3) )
    pinfo(f'Frequency: major {majorNum}, bg {bgNum}, blank {blankNum}')
    numArr = np.array([majorNum, bgNum, blankNum])
    md = np.median(numArr)
    weights = md/numArr
    pinfo(f'Weight array = {weights}')

    ### outputs
    pinfo('Configuring output paths')
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    weights_dir = cwd.parent.parent.joinpath('weights')

    model_weights = weights_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    input_shape = (X.shape[1], X.shape[2], 1)
    architecture = U_Net_Like(input_shape, 3)
    model = architecture.get_model()
    print(model.summary())

    # setup loss
    cce = categorical_focal_loss(alpha=weights, gamma=2)

    # setup metric
    ca = top2_categorical_accuracy

    # setup optimizer
    adam = Adam(1e-4)

    # setup callback
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)
    LRCallback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=3, verbose=0, mode='auto',
        min_delta=1e-6, cooldown=0, min_lr=1e-7
    )
    # print(cnn.summary())
    model.compile(optimizer=adam,\
                metrics = ca,\
                loss=cce)

    model.fit(x=X, y=Y,\
            validation_split=0.28,\
            shuffle=True,\
            batch_size=1, epochs=40,\
            callbacks = [CsvCallback, LRCallback])

    model.save(model_weights)

    pcheck_point('Finished Training')
    return C


if __name__ == "__main__":
    pbanner()
    psystem('Photographic track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    # load pickle
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('photographic.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    model_name = "photographic_mc_small_01"
    record_name = "photographic_record_mc_small_01"

    # setup parameters
    C.set_outputs(model_name, record_name)
    photographic_train(C)
