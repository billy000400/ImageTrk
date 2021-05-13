# Detector (Faster RCNN)
# forward propogate from input to output
# Goal: test if the validation output act as expected

import sys
from pathlib import Path
import pickle
import timeit
from datetime import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, TimeDistributed, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy

script_dir = Path.cwd().parent.parent.parent.joinpath('frcnn_mc_train')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config
from Layers import *
from Loss import *
from Metric import *


# load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

# re-build model
oneHotEncoder = C.oneHotEncoder
classNum = len(oneHotEncoder)
img_input = Input(shape=C.input_shape, name='x1')
RoI_input = Input(shape=(None, 4), name='x2')
x = C.base_net.get_base_net(img_input, trainable=False)
x = RoIPooling(6,6)([x, RoI_input])
x = TimeDistributed(Flatten(name='flatten'))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fcl', trainable=False))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fc2', trainable=False))(x)
x1 = TimeDistributed(Dense(classNum, name='fc3', trainable=False))(x)
x2 = TimeDistributed(Dense(classNum*4, activation='relu', name='fc4', trainable=False))(x)
output_classifier = TimeDistributed(Softmax(axis=-1), name='detector_out_class')(x1)
output_regressor = TimeDistributed(Dense(classNum*4, activation='linear', trainable=False), name='detector_out_regr')(x2)

model = Model(inputs={'x1':img_input, 'x2':RoI_input}, outputs = [output_classifier, output_regressor])
model.summary()

# load model weights
model.load_weights(str(Path.cwd().joinpath('rpn_mc_00.h5')), by_name=True)
model.load_weights(str(Path.cwd().joinpath('detector_mc_RCNN_dr=0.0.h5')), by_name=True)

# construct data generator
def gen_fn(X1_dir, X2_dir, batch_size=1):
    X1_list = [child for child in X1_dir.iterdir()]
    X2_list = [child for child in X2_dir.iterdir()]
    XX_list = list(zip(X1_list, X2_list))

    indexes_all = np.arange(len(XX_list))

    iterNum = int(np.floor(len(XX_list) / batch_size))

    for index in range(iterNum):
        indexes = indexes_all[index*batch_size:(index+1)*batch_size]
        XX_file_list = [XX_list[k] for k in indexes]

        X1_file_list = []
        X2_file_list = []
        for t in XX_file_list:
            X1_file_list.append(t[0])
            X2_file_list.append(t[1])

        X1 = [np.load(x_file) for x_file in X1_file_list]
        X2 = [np.load(x_file) for x_file in X2_file_list]
        X1 = np.array(X1, np.float32)
        X2 = np.array(X2, np.float32)
        yield [X1, X2]

gen = gen_fn(C.validation_img_inputs_npy, C.validation_rois)

# compile model
adam = Adam()
detector_class_loss = define_detector_class_loss(1)
detector_regr_loss = define_detector_regr_loss(100)
ca = CategoricalAccuracy()
model.compile(optimizer=adam, loss={'detector_out_class':detector_class_loss,\
                                    'detector_out_regr':detector_regr_loss},\
                                metrics = {'detector_out_class':ca,\
                                            'detector_out_regr':unmasked_IoUV2})


# prepare input data for prediction
iterNum = len([f for f in C.validation_rois.glob('*')])
df_r = pd.read_csv(C.validation_bbox_reference_file, index_col=None)
df_p = pd.read_csv(C.validation_bbox_proposal_file, index_col=None)
imgNames = df_p['FileName'].unique().tolist()
colNames = list(df_r.columns.values)
colName_dict = {cn:[] for i,cn in enumerate(colNames) if i>0}

# prepare output data frame
df_o1 = pd.DataFrame.from_dict(colName_dict)
df_o2 = pd.DataFrame.from_dict(colName_dict)

# prepare convenient functions
def delta_to_box(d):
    # proposal t = (x, y, w, h) as indicated in the original paper
    # (x,y) is the left upper corner
    p = [d[0], d[0]+d[2], d[1]-d[3], d[1]]
    if p[0] < 0:
        p[0] = 0
    if p[1] > 1:
        p[1] = 1
    if p[2] < 0:
        p[2] = 0
    if p[3] > 0:
        p[3] = 1
    return p


# manually iterate to predict by frcnn
for i in range(iterNum):
    sys.stdout.write(t_info(f'Processing image {i+1}/{iterNum}', '\r'))
    if i+1 == iterNum:
        sys.stdout.write('\n')
    sys.stdout.flush()

    imgName = imgNames[i]
    proposals = df_p[df_p['FileName']==imgName][['XMin', 'XMax', 'YMin', 'YMax']].to_numpy()
    x = next(gen)

    labels, deltas = model.predict_on_batch(x)
    labels = np.squeeze(labels, axis=0)
    deltas = np.squeeze(deltas, axis=0)
    deltas = np.reshape(deltas, (deltas.shape[0],-1, 4))

    maxIndexes = np.argmax(labels, axis=1)
    bg_mask = maxIndexes!=0 # only want bg_score < 0.5 proposals
    proposals = proposals[bg_mask]

    ### calculate non-bg score
    selected_labels = labels[bg_mask]
    scores = 1-selected_labels[:,0]

    deltas_raws = deltas[bg_mask]
    oj_indexes = np.argmax(selected_labels, axis=1)
    deltas = []
    for j,oj_index in enumerate(oj_indexes):
        deltas.append(deltas_raws[j,oj_index,:])
    deltas = np.array(deltas, dtype=np.float32)


    for proposal, score, delta in zip(proposals, scores, deltas):
        o1 = {'FileName':imgName,\
            'XMin':proposal[0], 'XMax':proposal[1],\
            'YMin':proposal[2], 'YMax':proposal[3],\
            'Score':score,\
            'ClassName':11}

        proposal = delta_to_box(delta)
        o2 = {'FileName':imgName,\
            'XMin':proposal[0], 'XMax':proposal[1],\
            'YMin':proposal[2], 'YMax':proposal[3],\
            'Score':score,\
            'ClassName':11}

        df_o1 = df_o1.append(o1, ignore_index=True)
        df_o2 = df_o2.append(o2, ignore_index=True)

df_o1.to_csv('rpn+detector_cls_prediction.csv')
df_o2.to_csv('rpn+detector_prediction.csv')
