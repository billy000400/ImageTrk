import sys
from pathlib import Path
import pickle
import csv

import numpy as np
import matplotlib.pyplot as plt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from mu2e_output import *

def hits_in_bbox_statistics(C):

    pinfo("Loading inputs from csv")
    ### inputs
    with open(C.X_file) as f1:
        f1_reader = csv.reader(f1)
        X = list(f1_reader)

    with open(C.Y_file) as f2:
        f2_reader = csv.reader(f2)
        Y = list(f2_reader)

    hit_nums = [len(row) for row in Y]


    positive_hits = [ [x for x in row if x == 'True'] for row in Y]
    pos_nums = [len(row) for row in positive_hits]

    negative_hits = [ [x for x in row if x == 'False'] for row in Y]
    neg_nums = [len(row) for row in negative_hits]

    fig1, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.hist(hit_nums, 100)
    ax1.set_title('Hit Number in Bounding Box Statistics')
    ax1.set_xlabel('Number of hits in bbox')
    ax1.set_ylabel('Number of cases')

    ax2.hist(pos_nums, 100)
    ax2.set_title('Positive Hit Number in Bounding Box Statistics')
    ax2.set_xlabel('Number of positive hits in bbox')
    ax2.set_ylabel('Number of cases')

    ax3.hist(neg_nums, 100)
    ax3.set_title('Negative Hit Number in Bounding Box Statistics')
    ax3.set_xlabel('Number of negative hits in bbox')
    ax3.set_ylabel('Number of cases')

    plt.show()



if __name__ == "__main__":
    pbanner()
    psystem('CNN track extractor')
    pmode('Training')

    # load pickle
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    hits_in_bbox_statistics(C)
