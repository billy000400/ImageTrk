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
from tensorflow.keras import Model
from tensorflow.keras.layers import(
    Input,
    Dense,
    Conv2D,
    MaxPool2D,Dropout,
    Flatten,
    TimeDistributed,
    Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *
from frcnn_loss import *
from frcnn_metric import *
### import ends

def predict(C):

    pstage('Start predicting tracks')

    ## unpacking parameters from configuration
    pinfo("Configuring I/O path")
    cwd = Path.cwd()
    weights_dir = cwd.parent.parent.joinpath('weights')
    model_weights = weights_dir.joinpath(C.model_name+'.h5')

    data_dir = C.test_Y_file_reference.parent
    Y_name = "hit_truth_test_prediction.csv"
    Y_file = data_dir.joinpath(Y_name)
    Y_open = open(Y_file, 'w')
    Ywriter = csv.writer(Y_open, delimiter=',')

    ### set inputs
    # load raw list from csv
    pinfo("Loading the testing input from csv")
    with open(C.test_X_file) as f1:
        f1_reader = csv.reader(f1)
        X_raw = list(f1_reader)

    # split rows having > 2000 positions into pieces
    X = []
    bboxNum = len(X_raw)
    for idx, row in enumerate(X_raw):
        sys.stdout.write(t_info(f'Examining bbox {idx+1}/{bboxNum}', '\r'))
        if (idx+1)==bboxNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        if len(row) <= 3*C.sequence_max_length:
            X.append(row)
        else:
            X_last = row
            while len(X_last) > 3*C.sequence_max_length:
                X.append(X_last[:3*C.sequence_max_length])
                X_last = X_last[3*C.sequence_max_length:]


    ### padding inputs
    pinfo("Padding input sequences")
    X = sequence.pad_sequences(X, maxlen=3*C.sequence_max_length,\
                        dtype='float32',padding='post',value=0.0)
    X = np.reshape(X,(len(X), -1, 3))

    ### reconstruct model
    # recur model
    pinfo("Assembling the CNN extractor")
    input_layer = Input(shape=(X.shape[1], X.shape[2], 1))
    x = Conv2D(32, (10, 3), padding='same', activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    output_layer = Dense(C.sequence_max_length, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    # load weights
    model.load_weights(model_weights, by_name=True)

    ### make predictions
    # get raw outputs from model prediction
    pinfo("The CNN track extractor is extracting tracks from bboxes")
    outputs_raw = model.predict(x=X, batch_size=1)
    # The raw output is enough for evaluating the model
    outputs_raw = outputs_raw
    outputs_raw[outputs_raw<0.5] = False
    outputs_raw[outputs_raw>=0.5] = True
    outputs_raw = np.asarray(outputs_raw, dtype='bool')
    outputs_raw = outputs_raw.tolist()
    for output in outputs_raw:
        Ywriter.writerow(output)

    ### record the prediction file path
    C.set_prediction(Y_file)

    ### save pickle
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.test.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pcheck_point('Predicted hit truths')

    return C

if __name__ == "__main__":
    pbanner()
    psystem('CNN track extractor')
    pmode('Testing')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.test.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    C = predict(C)
