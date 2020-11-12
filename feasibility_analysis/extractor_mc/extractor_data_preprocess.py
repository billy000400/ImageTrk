import sys
import csv
from pathlib import Path
import random
import pickle

import numpy as np

from tensorflow.keras.preprocessing import sequence

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *

def preprocess(C):
    pstage("Preprocessing data")

    # load lists
    pinfo("Loading inputs from csv")
    ### inputs
    with open(C.X_file) as f1:
        f1_reader = csv.reader(f1)
        X_raw = list(f1_reader)

    with open(C.Y_file) as f2:
        f2_reader = csv.reader(f2)
        Y_raw = list(f2_reader)

    # split rows having > C.sequence_max_length positions into pieces
    X = []
    Y = []
    bboxNum = len(Y_raw)
    for idx, row in enumerate(Y_raw):
        sys.stdout.write(t_info(f'Examining bbox {idx+1}/{bboxNum}', '\r'))
        if (idx+1)==bboxNum:
            sys.stdout.write('\n')
        sys.stdout.flush()


        if len(row) <= C.sequence_max_length:
            X.append(X_raw[idx])
            Y.append(row)
        else:
            X_last = X_raw[idx]
            Y_last = row
            while len(Y_last) > C.sequence_max_length:
                X.append(X_last[:3*C.sequence_max_length])
                Y.append(Y_last[:C.sequence_max_length])
                X_last = X_last[3*C.sequence_max_length:]
                Y_last = Y_last[C.sequence_max_length:]

    Y = [ [el=="True" for el in row ] for row in Y ]


    # make a reference of true False(due to bg rather than padding) hits
    true_False_ref = []
    for row in Y:
        False_indices = [i for i,truth in enumerate(row) if truth == False]
        true_False_ref.append(False_indices)

    ### reshape input positions from flat to 2D
    pinfo("Padding input sequences")
    # pad tracks with 0's
    X = sequence.pad_sequences(X, maxlen=3*C.sequence_max_length,\
                        dtype='float32',padding='post',value=0.0)
    Y = sequence.pad_sequences(Y, maxlen=C.sequence_max_length,\
                        dtype='bool',padding='post',value=False)
    X = np.reshape(X,(len(X), -1, 3))

    ### mask entries to make numbers of True and False equivalent
    pinfo("Masking entries for unbiased training")

    mY = np.zeros(shape=Y.shape)

    bboxNum = Y.shape[0]
    for i in range(bboxNum):
        sys.stdout.write(t_info(f'Masking hits in bbox {i+1}/{bboxNum}', '\r'))
        if (i+1)==bboxNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        y = Y[i]
        my = np.zeros(shape=y.shape)
        my[:] = np.nan

        True_indices = np.where(y==True)[0]
        true_False_indices = true_False_ref[i]
        pad_False_indices = [ index for index in list(range(C.sequence_max_length)) if (index not in True_indices) and (index not in true_False_indices) ]


        my[y==True] = True

        posNum = np.count_nonzero(y==True)
        negNum = len(true_False_indices)+len(pad_False_indices)
        if posNum<negNum:
            negWant = posNum
            true_False_want = posNum
            pad_False_want = 0
            if len(true_False_indices) < true_False_want:
                true_False_want = len(true_False_indices)
                pad_False_want = negWant-true_False_want

            true_False_index_selected_list = random.sample(true_False_indices, true_False_want)
            pad_False_index_selected_list = random.sample(pad_False_indices, pad_False_want)
            neg_index_selected_list = true_False_index_selected_list+pad_False_index_selected_list
        else:
            neg_index_selected_list = true_False_indices+pad_False_indices

        for index in neg_index_selected_list:
            my[index] = False

        mY[i]=my

    ### save numpy arrays to tmp
    pinfo("Saving numpy arrays to local")
    cwd = Path.cwd()
    tmp_dir = cwd.parent.parent.joinpath('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np_dir = tmp_dir

    X_npy = np_dir.joinpath('extractor_train_X.npy')
    mY_npy = np_dir.joinpath('extractor_train_masked_Y.npy')

    np.save(X_npy, X)
    np.save(mY_npy, mY)

    # setup configuration
    C.set_input_array(X_npy, mY_npy)

    pickle_train_path = Path.cwd().joinpath('extractor.train.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pcheck_point("Processed numpy arrays")
    return C

if __name__ == "__main__":
    pbanner()
    psystem('CNN track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    # load pickle
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    sequence_max_length = 1000

    # setup parameters
    C.set_max_length(sequence_max_length)
    preprocess(C)
