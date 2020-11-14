import sys
import csv
from pathlib import Path
import random
import pickle

import numpy as np

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *

def preprocess(C):
    pstage("Preprocessing data")

    # load lists
    pinfo("Loading raw numpy arrays")
    X = np.load(C.X_file)
    Y = np.load(C.Y_file)

    ### pixel truth labels
    is_blank = np.array([1,0,0], dtype=bool)
    is_bg = np.array([0,1,0], dtype=bool)
    is_major = np.array([0,0,1], dtype=bool)

    ### mask entries to make numbers of True and False equivalent
    pinfo("Masking entries for unbiased training")

    mY = np.zeros(shape=Y.shape, dtype=bool)

    bboxNum = Y.shape[0]
    for i in range(bboxNum):
        sys.stdout.write(t_info(f'Masking hits in bbox {i+1}/{bboxNum}', '\r'))
        if (i+1)==bboxNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        y = Y[i]
        my = np.zeros(shape=y.shape, dtype=np.bool)
        my[:] = np.nan

        blank_indices = np.where(y[:,:,0]==True)
        bg_indices = np.where(y[:,:,1]==True)
        major_indices = np.where(y[:,:,2]==True)

        my[y[:,:,2]==True] = is_major

        posNum = major_indices[0].size
        negNum = bg_indices[0].size

        if posNum<negNum:
            negWant = posNum

            range_list = list(range(negWant))
            neg_index_selected_list = random.sample(range_list,negWant)

            neg_index_selected_arr = np.array(neg_index_selected_list)
            row_selected_arr = bg_indices[0][neg_index_selected_arr]
            col_selected_arr = bg_indices[1][neg_index_selected_arr]
            bg_indices = (row_selected_arr, col_selected_arr)



        my[bg_indices] = is_bg

        blankWant = posNum

        blank_range_list = list(range(blankWant))
        blank_index_selected_list = random.sample(blank_range_list,blankWant)

        blank_index_selected_arr = np.array(blank_index_selected_list)
        try:
            blank_row_selected_arr = blank_indices[0][blank_index_selected_arr]
        except:
            pdebug(posNum)
            pdebug(blank_index_selected_arr)
        blank_col_selected_arr = blank_indices[1][blank_index_selected_arr]
        blank_indices = (blank_row_selected_arr, blank_col_selected_arr)

        my[blank_indices] = is_blank

        mY[i]=my

    ### save numpy arrays to tmp
    pinfo("Saving numpy arrays to local")
    cwd = Path.cwd()
    tmp_dir = cwd.parent.parent.joinpath('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np_dir = tmp_dir

    X_npy = np_dir.joinpath('photographic_train_X.npy')
    mY_npy = np_dir.joinpath('photographic_train_masked_Y.npy')

    np.save(X_npy, X)
    np.save(mY_npy, mY)

    # setup configuration
    C.set_input_array(X_npy, mY_npy)

    pickle_train_path = Path.cwd().joinpath('photographic.train.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pcheck_point("Processed numpy arrays")
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

    preprocess(C)
