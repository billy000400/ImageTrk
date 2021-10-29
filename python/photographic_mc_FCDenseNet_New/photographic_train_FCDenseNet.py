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


util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config
from DataGenerator import DataGenerator as Generator
from Architectures import FC_DenseNet
from Information import *
from Loss import *
from Metric import *
### import end

print(tf.executing_eagerly())

def photographic_train(C):
    pstage("Start Training")

    ### outputs
    pinfo('Configuring output paths')
    cwd = Path.cwd()
    data_dir = C.sub_data_dir
    weights_dir = C.data_dir.parent.joinpath('weights')

    model_weights_file = weights_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    input_shape = (C.resolution, C.resolution, 1)
    architecture = FC_DenseNet(input_shape, 3, dr=0.1)
    model = architecture.get_model()
    model.summary()

    # setup loss
    weights = C.weights
    assert weights.all()!=None, "The weight is None! You need to calculate the weights first!"
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

    earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_top2_categorical_accuracy', patience=10)

    ModelCallback = tf.keras.callbacks.ModelCheckpoint(model_weights_file,\
                        monitor='val_top2_categorical_accuracy', verbose=1,\
                        save_weights_only=True,\
                        save_best_only=True, mode='auto', save_freq='epoch')

    logdir="logs/fit/" + C.model_name+'_' + datetime.now().strftime("%Y%m%d-%H%M%S")
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
            callbacks = [CsvCallback,earlyStopCallback, ModelCallback, tensorboard_callback],\
            epochs=150)
    model.save(model_weights_file)

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
    model_name = "photographic_mc_arc_FCDense_Dropout_0.1_Unaugmented"
    record_name = "photographic_record_mc_arc_FCDense_Dropout_0.1_UnAugmented"

    # setup parameters
    C.set_outputs(model_name, record_name)
    photographic_train(C)
