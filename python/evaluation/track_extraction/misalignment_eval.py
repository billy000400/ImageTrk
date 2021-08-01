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
model.compile(optimizer=adam, loss=cce, metrics=[ca, hit_purity, hit_efficiency])

def data_generator(xDir, yDir, iouDir, ratioDir):
    xlist = [f for f in xDir.iterdir()]
    ylist = [f for f in yDir.iterdir()]
    ioulist = [f for f in iouDir.iterdir()]
    ratiolist = [f for f in ratioDir.iterdir()]

    for xf,yf,iouf,ratiof in zip(xlist,ylist,ioulist,ratiolist):
        x = np.expand_dims(np.load(xf),axis=0)
        y = np.expand_dims(np.load(yf), axis=0)
        iouv = pickle.load(open(iouf, 'rb'))
        ratio = pickle.load(open(ratiof, 'rb'))
        yield x,y,iouv, ratio

gen = data_generator(C.X_train_dir, C.Y_train_dir, C.iou_dir,\
                        C.sub_data_dir.joinpath('misaligned_hit_ratios'))

listDir = Path.cwd().joinpath('list')
listDir.mkdir(exist_ok=True)
iou_f = listDir.joinpath('iou')
acc_f = listDir.joinpath('acc')
pur_f = listDir.joinpath('pur')
eff_f = listDir.joinpath('eff')
ratio_f = listDir.joinpath('ratio')

if iou_f.exists():
    ious = pickle.load(open(iou_f, 'rb'))
    accs = pickle.load(open(acc_f, 'rb'))
    purs = pickle.load(open(pur_f, 'rb'))
    effs = pickle.load(open(eff_f, 'rb'))
    ratios = pickle.load(open(ratio_f,'rb'))
else:

    ious, accs, purs, effs, ratios = [], [], [], [], []
    for i in range(sampleNum):
        sys.stdout.write(t_info(f'Processing image {i+1}/{sampleNum}', '\r'))
        if i+1 == sampleNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        x, y, iou_val, ratio = next(gen)
        ious.append(iou_val)
        ratios.append(ratio)

        result = model.test_on_batch(x, y, return_dict=True)
        accs.append(result['top2_categorical_accuracy'])
        purs.append(result['hit_purity'])
        effs.append(result['hit_efficiency'])

    pickle.dump(ious, open(iou_f,'wb'))
    pickle.dump(accs, open(acc_f,'wb'))
    pickle.dump(purs, open(pur_f,'wb'))
    pickle.dump(effs, open(eff_f,'wb'))
    pickle.dump(ratios, open(ratio_f,'wb'))


plt.scatter(ious, accs, s=.5, alpha=.3)
plt.xlabel('IoU')
plt.ylabel('top 2 categorical accuracy')
plt.show()

plt.scatter(ious, purs, s=.5, alpha=.3)
plt.xlabel('IoU')
plt.ylabel('hit purity')
plt.show()

plt.scatter(ious, effs, s=.5, alpha=.3)
plt.xlabel('IoU')
plt.ylabel('hit efficiency')
plt.show()

plt.scatter(ious, ratios, s=.5, alpha=.3)
plt.xlabel('IoU')
plt.ylabel('Remained Hit Ratio')
plt.show()

bins = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.0]
result_acc = binning_objects(accs,ious,bins) # binned average accuracy
result_pur = binning_objects(purs,ious,bins) # binned hit purity
result_eff = binning_objects(effs,ious,bins) # binned hit efficiency
result_ratio = binning_objects(ratios,ious,bins) # binned hit ratio

print(f'Total amount of data: {len(ious)}')
print(f'Number of bins: {len(result_acc)}')
sum_val = 0
for index, x in enumerate(result_acc):
    if len(x)==0:
        print(f"bin {index} is empty")
    else:
        print(f"bin {index} has {len(x)} values")
        sum_val += len(x)
print(f'Total amounut of binned data {sum_val}')

result_acc = result_acc[1:]
result_pur = result_pur[1:]
result_eff = result_eff[1:]
result_ratio = result_ratio[1:]


def mean(vals):
    if len(vals)==0:
        return None
    else:
        print(sum(vals))
        return sum(vals)/float(len(vals))

acc_avgs= [mean(vals) for vals in result_acc]
pur_avgs= [mean(vals) for vals in result_pur]
eff_avgs= [mean(vals) for vals in result_eff]
ratio_avgs = [mean(vals) for vals in result_ratio]

acc_stds = [np.array(vals).std() for vals in result_acc]
pur_stds = [np.array(vals).std() for vals in result_pur]
eff_stds = [np.array(vals).std() for vals in result_eff]
ratio_stds = [np.array(ratios).std() for vals in result_ratio]

mids = [ (bins[i]+bins[i+1])/2 for i in range(len(bins)-1) ]

plt.plot(mids, acc_avgs)
plt.xlabel('IoU')
plt.ylabel('top 2 categorical accuracy')
plt.show()

plt.plot(mids, pur_avgs)
plt.xlabel('IoU')
plt.ylabel('average hit purity')
plt.show()

plt.plot(mids, eff_avgs)
plt.xlabel('IoU')
plt.ylabel('average hit efficiency')
plt.show()

plt.plot(mids, ratio_avgs)
plt.title('Remained Hit Ratio vs. IoU')
plt.xlabel('IoU')
plt.ylabel('Remained Hit Ratio')
plt.show()

plt.errorbar(mids, acc_avgs, yerr=acc_stds, label='top 2 ca', capsize=5)
plt.errorbar(mids, pur_avgs, yerr=pur_stds, label='hit purity', capsize=10)
plt.errorbar(mids, eff_avgs, yerr=eff_stds, label='hit efficiency', capsize=15)
plt.xlabel('IoU')
plt.ylabel('%')
plt.title('The Misalignment Effect on Track Extraction')
plt.legend()
plt.show()

print(acc_avgs)
