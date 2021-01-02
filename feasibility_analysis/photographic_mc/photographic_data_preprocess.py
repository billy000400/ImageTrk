import sys
import csv
from pathlib import Path
import random
import pickle
from copy import deepcopy
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Config import extractor_config as Config
from mu2e_output import *

def preprocess(C):
    pstage("Preprocessing data")

    # load lists
    pinfo("Loading raw numpy arrays")
    X = np.load(C.X_file, allow_pickle=True)
    Y = np.load(C.Y_file, allow_pickle=True)

    # calculate scaled dimensions
    res = C.resolution
    scale_want = (res, int(res/3*2)) # notice that this is the scale for IMAGES!
    scale_alter = (int(res/3*2), res)

    ### pixel truth labels
    is_blank = np.array([1,0,0], dtype=np.float32)
    is_bg = np.array([0,1,0], dtype=np.float32)
    is_major = np.array([0,0,1], dtype=np.float32)

    ### mask entries to make numbers of True and False equivalent
    pinfo("Masking entries for unbiased training")

    bboxNum = len(Y)
    sX_shape = (bboxNum, scale_want[1], scale_want[0])
    sX = np.zeros(shape=sX_shape, dtype=np.float32)
    mY_shape = (bboxNum, scale_want[1], scale_want[0], 3)
    mY = np.zeros(shape=mY_shape, dtype=np.float32)
    for i in range(bboxNum):
        sys.stdout.write(t_info(f'Masking hits in bbox {i+1}/{bboxNum}', '\r'))
        if (i+1)==bboxNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # scale the input and output
        xo=deepcopy(X[i]).astype(np.uint8) # convert to int8 for PIL's RGB interperation
        yo=deepcopy(Y[i]).astype(np.uint8)

        ratio = xo.shape[0]/xo.shape[1]
        im_in = Image.fromarray(xo)
        im_out = Image.fromarray(yo, mode='RGB')

        if ratio < 1:
            x = im_in.resize(scale_want)
            y = im_out.resize(scale_want)
        else:
            x = im_in.resize(scale_alter)
            y = im_out.resize(scale_alter)
            x = x.rotate(angle=90, expand=True)
            y = y.rotate(angle=90, expand=True)

        x = np.array(x, dtype=np.float32)# back to float to be masked by nan
        y = np.array(y, dtype=np.float32)

        # prepare masked data
        my = np.zeros(shape=y.shape, dtype=np.float32)
        my[:] = np.nan

        blank_indices = np.where(y[:,:,0]==1)
        bg_indices = np.where(y[:,:,1]==1)
        major_indices = np.where(y[:,:,2]==1)

        my[y[:,:,2]==1] = is_major

        majorNum = major_indices[0].size
        bgNum = bg_indices[0].size
        blankNum = blank_indices[0].size

        if majorNum<bgNum:
            bgWant = majorNum

            range_list = list(range(bgNum))
            bg_index_selected_list = random.sample(range_list,bgWant)

            bg_index_selected_arr = np.array(bg_index_selected_list)
            try:
                bg_row_selected_arr = bg_indices[0][bg_index_selected_arr]
            except:
                plt.imshow(x)
                plt.show()
                sys.exit()
            bg_col_selected_arr = bg_indices[1][bg_index_selected_arr]
            bg_indices = (bg_row_selected_arr, bg_col_selected_arr)



        my[bg_indices] = is_bg

        blankWant = majorNum

        blank_range_list = list(range(blankNum))
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

        sX[i]=x
        mY[i]=my

    ### save numpy arrays to tmp
    pinfo("Saving numpy arrays to local")
    cwd = Path.cwd()
    tmp_dir = cwd.parent.parent.joinpath('tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np_dir = tmp_dir

    sX_npy = np_dir.joinpath('photographic_train_X.npy')
    mY_npy = np_dir.joinpath('photographic_train_masked_Y.npy')

    np.save(sX_npy, sX)
    np.save(mY_npy, mY)

    # setup configuration
    C.set_input_array(sX_npy, mY_npy)

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
