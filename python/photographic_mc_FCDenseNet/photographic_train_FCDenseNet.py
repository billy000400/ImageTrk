# This script's purpose is to train a preliminary CNN for tracking by Keras
# Author: Billy Li
# Email: li000400@umn.edu

# import starts
import sys
from pathlib import Path
import csv
import random
import pickle
from datetime import datetime

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
from DataGenerator import DataGenerator as Generator
from Architecture import FC_DenseNet
from mu2e_output import *
from Loss import *
from Metric import *
### import ends

def photographic_train(C):
    pstage("Start Training")

    ### outputs
    pinfo('Configuring output paths')
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    weights_dir = cwd.parent.parent.joinpath('weights')

    model_weights = weights_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    input_shape = (C.resolution, C.resolution, 1)
    architecture = FC_DenseNet(input_shape, 3, dr=0.2)
    model = architecture.get_model()
    model.summary()

    # setup loss
    weights = [1, 1.62058132, 1.13653004e-03]
    cce = categorical_focal_loss(alpha=weights, gamma=2)

    # setup metric
    ca = top2_categorical_accuracy

    # setup optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # setup callback
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # print(cnn.summary())
    model.compile(optimizer=adam,\
                metrics = ca,\
                loss=cce)



    train_generator = Generator(C.X_train_dir, C.Y_train_dir, batch_size=1)
    val_generator = Generator(C.X_val_dir, C.Y_val_dir, batch_size=1)

    model.fit(x=train_generator,\
            shuffle=True,\
            validation_data=val_generator,\
            callbacks = [CsvCallback, tensorboard_callback],\
            use_multiprocessing=True,\
            workers=4,\
            epochs=50)
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
    model_name = "photographic_mc_arc_FCDense_Dropout_0.3"
    record_name = "photographic_record_mc_arc_FCDense_Dropout_0.3"

    # setup parameters
    C.set_outputs(model_name, record_name)
    photographic_train(C)
