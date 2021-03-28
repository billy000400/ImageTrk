### This file illustrates the number of proposals per image with the nature
# of those proposals {True True? False True? Duplicated?}.
# The configuration pickle should not be written to the disk

import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Information import *

cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

df_r = pd.read_csv(C.bbox_reference_file, index_col=0) # r=real
df_p = pd.read_csv(C.bbox_proposal_file, index_col=0) # p=proposed

imgNames = df_r['FileName'].unique().tolist()
proposalNum = []
for img in imgNames:
    proposalNum.append(len(df_p[df_p['FileName']==img].index))

proposalNum = np.array(proposalNum)
plt.hist(proposalNum)
plt.show()
print(proposalNum.min())
