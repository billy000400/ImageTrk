### imports starts
import sys
from pathlib import Path
import pickle
import csv

import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing import sequence

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_util import *
from mu2e_output import *
from extractor_config import Config
### import ends

def extractor_test(C):

    ### load reference Y and prediction Y
    pinfo("Loading reference and prediction results")
    with open(C.test_Y_file_reference) as f1:
        f1_reader = csv.reader(f1)
        Y_ref = list(f1_reader)

    with open(C.test_Y_file_prediction) as f2:
        f2_reader = csv.reader(f2)
        Y_pred = list(f2_reader)


    Y_ref = np.array(Y_ref)
    Y_ref = sequence.pad_sequences(Y_ref, maxlen=C.sequence_max_length,\
                        dtype='bool',padding='post',value=False)

    Y_pred = [ [el=='True' for el in row] for row in Y_pred  ]
    Y_pred = np.array(Y_pred)

    pos_pred_indices = Y_pred==True
    proposed_positive = np.count_nonzero(Y_pred)
    true_positive = np.count_nonzero(Y_ref[pos_pred_indices])
    precision = true_positive/proposed_positive


    all_positive = np.count_nonzero(Y_ref)
    recall = true_positive/all_positive

    score = np.count_nonzero(np.equal(Y_ref, Y_pred))/Y_pred.size
    pinfo(f"Average binary accuracy: {score}")
    pinfo(f"Precision: {precision}")
    pinfo(f"Recall: {recall}")

    file = open("extractor_test_result.csv", "w+")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"Average binary accuracy: {score}\n")

if __name__ == "__main__":
    pbanner()
    psystem('CNN track extractor')
    pmode('Testing')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.test.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    extractor_test(C)
