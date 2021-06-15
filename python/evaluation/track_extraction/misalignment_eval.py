### This script is to test the sensitivity of the extractor to misalignment
# the misalignment is caused by imperfect track detection (IoU != 100%)

import sys
import csv
from pathlib import Path
import shutil
from collections import Counter
import pickle
from copy import deepcopy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

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

script_dir = Path.cwd().parent.parent.joinpath('photographic_mc_FCDenseNet')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config
from Abstract import binning_objects
from Architectures import FC_DenseNet
from Geometry import*
from Database import *
from Information import *
from Loss import *
from Metric import *

# load configuration object
cwd = Path.cwd()

pickle_path = cwd.joinpath('extractor_perfect_misalignment.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

foo_list = [f for f in C.X_train_dir.glob('*')]
sampleNum = len(foo_list)

# re-build model
input_shape = (C.resolution, C.resolution, 1)
architecture = FC_DenseNet(input_shape, 3, dr=0.0)
model = architecture.get_model()
model.summary()

# load model weights
weight_file_name = 'photographic_mc_arc_FCDense_Dropout_0.1_Unaugmented.h5'
model.load_weights(str(Path.cwd().joinpath(weight_file_name)), by_name=True)

# evaluate model
weights = [1,1,1]
cce = categorical_focal_loss(alpha=weights, gamma=2)
ca = top2_categorical_accuracy
adam = Adam()
model.compile(optimizer=adam, loss=cce, metrics=ca)

def data_generator(xdir, ydir, ioudir):
    xlist = [f for f in xdir.iterdir()]
    ylist = [f for f in ydir.iterdir()]
    ioulist = [f for f in ioudir.iterdir()]

    for xf,yf,iouf in zip(xlist,ylist,ioulist):
        x = np.expand_dims(np.load(xf),axis=0)
        y = np.expand_dims(np.load(yf), axis=0)
        iouv = pickle.load(open(iouf, 'rb'))
        yield x,y,iouv

gen = data_generator(C.X_train_dir, C.Y_train_dir, C.iou_dir)

xs, ys = [], []
for i in range(sampleNum):
    sys.stdout.write(t_info(f'Processing image {i+1}/{sampleNum}', '\r'))
    if i+1 == sampleNum:
        sys.stdout.write('\n')
    sys.stdout.flush()

    x, y, iou_val = next(gen)
    xs.append(iou_val)

    result = model.test_on_batch(x, y, return_dict=True)
    ys.append(result['top2_categorical_accuracy'])

plt.scatter(xs, ys, s=.5, alpha=.3)
plt.xlabel('IoU')
plt.ylabel('top 2 categorical accuracy')
plt.show()

bins = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.0]
result = binning_objects(ys,xs,bins)
print(f"There are {len(result[0])} ious less than .5")
result = result[1:]

print(len(xs))
for bin in result:
    print(len(bin))

def mean(vals):
    if len(vals)==0:
        return None
    else:
        return sum(vals)/float(len(vals))

averages = [mean(vals) for vals in result]

mids = [ (bins[i]+bins[i+1])/2 for i in range(len(bins)-1) ]

for index, x in enumerate(result):
    if len(x)==0:
        print(f"bin {index} is empty")
    else:
        print(f"bin {index} has {len(x)} values")

plt.plot(mids, averages)
plt.xlabel('IoU')
plt.ylabel('top 2 categorical accuracy')
plt.show()

print(ys)
print('mean average precision:',np.array(ys).mean())
