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

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import make_anchors
from Information import *
from Configuration import frcnn_config
from Layers import *

### Using a specific pair of CPU and GPU
# I pick the first GPU because it is faster
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.config.experimental.get_visible_devices())

### load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

### rebuild RPN
img_input = Input(shape=C.input_shape, name='x1')
x = C.base_net.get_base_net(img_input, trainable=False)
rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
rpn_cls = rpn_layer.classifier(x)
rpn_rgr = rpn_layer.regression(x)

model_rpn = Model(inputs=img_input, outputs=[x, rpn_cls, rpn_rgr])
model_rpn.load_weights(\
        str(C.weight_dir.joinpath(C.rpn_model_name+'.h5')), by_name=True)
model_rpn.load_weights(\
        str(C.weight_dir.joinpath(C.detector_model_name+'.h5')), by_name=True)

### build an RPN-to_RoI function
@tf.function(experimental_relax_shapes=True)
def delta_to_roi(acs, dts):
    txs, tys, tws, ths = tf.unstack(dts, axis=1)
    xmins, xmaxs, ymins, ymaxs = tf.unstack(acs, axis=1)
    xas = tf.math.divide(tf.math.add(xmins,xmaxs), 2.0) # center xs
    yas = tf.math.divide(tf.math.add(ymins,ymaxs), 2.0) # center ys
    was = tf.math.subtract(xmaxs,xmins) # widths
    has = tf.math.subtract(ymaxs,ymins) # heightss

    xs = tf.math.add(\
            tf.math.multiply(txs,was), xas)
    ys = tf.math.add(\
            tf.math.multiply(tys,has), yas)
    ws = tf.math.multiply(\
            tf.math.exp(tws), was)
    hs = tf.math.multiply(\
            tf.math.exp(ths), has)

    xmins = tf.math.subtract(xs, tf.math.divide(ws, 2.0))
    xmaxs = tf.math.add(xs, tf.math.divide(ws, 2.0))
    ymins = tf.math.subtract(ys, tf.math.divide(hs, 2.0))
    ymaxs = tf.math.add(ys, tf.math.divide(hs, 2.0))
    rois = tf.stack([ymaxs, xmins, ymins, xmaxs], axis=1)
    rois = tf.clip_by_value(t=rois, clip_value_min=0, clip_value_max=1)
    return rois


@tf.function(experimental_relax_shapes=True)
def diagonal_to_xywh(diagonals):
    # diagonal: (ymaxs, xmins, ymins, xmaxs)
    # xywh: (xmins, ymaxs, widths, heights)
    ymaxs, xmins, ymins, xmaxs = tf.unstack(diagonals, axis=1)
    widths = tf.math.subtract(xmaxs, xmins)
    heights = tf.math.subtract(ymaxs, ymins)
    return tf.stack([xmins, ymaxs, widths, heights], axis=1)

anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios)
acs = tf.constant(anchors, dtype=tf.float32)

labels_spec = tf.TensorSpec(shape=(32,32,18), dtype=tf.float32)
deltas_spec = tf.TensorSpec(shape=(32,32,72), dtype=tf.float32)
@tf.function(input_signature=[labels_spec, deltas_spec])
def rpn_to_roi(lbs, dts):
    h,w,d = lbs.shape
    dts_flat = tf.reshape(dts, shape=(h,w,d,4))
    indices = tf.where(tf.math.greater(lbs,0.5))
    scores = tf.gather_nd(params=lbs, indices=indices)
    acs_want = tf.gather_nd(params=acs, indices=indices)
    dts_want = tf.gather_nd(params=dts_flat, indices=indices)

    rois = delta_to_roi(acs_want, dts_want)

    selected_indices, selected_scores =\
    non_max_suppression_with_scores(rois, scores,\
                max_output_size=2000,\
                iou_threshold=0.7, score_threshold=0.0,\
                soft_nms_sigma=0.0)

    rois_want = tf.gather(params=rois, indices=selected_indices)
    rois_xywh = diagonal_to_xywh(rois_want)
    return rois_xywh, selected_scores

# rebuild detector
detector_fetch = Input(shape=(32, 32, 512), name='base_to_detector')
RoI_input = Input(shape=(None, 4), name='roi')
x = RoIPooling(6,6)([detector_fetch, RoI_input])
x = TimeDistributed(Flatten(name='flatten'))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fcl', trainable=False))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fc2', trainable=False))(x)
classNum = 2
x1 = TimeDistributed(Dense(classNum, name='fc3', trainable=False))(x)
x2 = TimeDistributed(Dense(classNum*4, activation='relu', name='fc4', trainable=False))(x)
output_classifier = TimeDistributed(Softmax(axis=-1), name='detector_out_class')(x1)
output_regressor = TimeDistributed(Dense(classNum*4, activation='linear', trainable=False), name='detector_out_regr')(x2)



model_detector = Model(inputs=[detector_fetch, RoI_input], outputs = [output_classifier, output_regressor])
model_detector.load_weights(\
        str(C.weight_dir.joinpath(C.detector_model_name+'.h5')), by_name=True)



# construct data generator
def gen_fn(dir, batch_size=1):
    f_list = [child for child in dir.iterdir()]

    indexes_all = np.arange(len(f_list))

    iterNum = int(np.floor(len(f_list) / batch_size))

    for index in range(iterNum):
        indexes = indexes_all[index*batch_size:(index+1)*batch_size]
        f_list_slice = [f_list[k] for k in indexes]

        arr = [np.load(f) for f in f_list_slice]
        yield np.array(arr, dtype=np.float32)

gen = gen_fn(C.validation_img_inputs_npy)

# prepare input data for prediction
iterNum = len([f for f in C.validation_img_inputs_npy.glob('*')])
df_r = pd.read_csv(C.validation_bbox_reference_file, index_col=None)
imgNames = df_r['FileName'].unique().tolist()
colNames = list(df_r.columns.values)
colName_dict = {cn:[] for i,cn in enumerate(colNames) if i>0}
colName_dict['Score'] = []
# prepare output data frame
df_o0 = pd.DataFrame.from_dict(colName_dict)
df_o1 = pd.DataFrame.from_dict(colName_dict)
df_o2 = pd.DataFrame.from_dict(colName_dict)


def delta_to_bbox(d):
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

@tf.function(experimental_relax_shapes=True)
def xywh_to_bbox(xywhs):
    # xywh: (xmins, ymaxs, widths, heights)
    # bbox: (xmins, xmaxs, ymins, ymaxs)
    xmins, ymaxs, widths, heights = tf.unstack(xywhs, axis=1)
    xmaxs = tf.math.add(xmins,widths)
    ymins = tf.math.subtract(ymaxs,heights)
    return tf.stack([xmins, xmaxs, ymins, ymaxs], axis=1)

# manually iterate to predict by frcnn
for i in range(iterNum):
    sys.stdout.write(t_info(f'Processing image {i+1}/{iterNum}', '\r'))
    if i+1 == iterNum:
        sys.stdout.write('\n')
    sys.stdout.flush()

    x = next(gen)

    imgName = imgNames[i]

    ftr_maps, rpn_score_maps, rpn_delta_maps = model_rpn.predict_on_batch(x)
    rpn_scores = np.squeeze(rpn_score_maps, axis=0)
    rpn_deltas = np.squeeze(rpn_delta_maps, axis=0)

    rois, rpn_scores = rpn_to_roi(rpn_scores, rpn_deltas)
    rpn_proposals = xywh_to_bbox(rois).numpy()
    rpn_scores = rpn_scores.numpy()

    for proposal, score in zip(rpn_proposals, rpn_scores):
        o0= {'FileName':imgName,\
            'XMin':proposal[0], 'XMax':proposal[1],\
            'YMin':proposal[2], 'YMax':proposal[3],\
            'Score':score}

        df_o0 = df_o0.append(o0, ignore_index=True)

    rois = np.expand_dims(rois, axis=0)
    detector_label_maps, detector_delta_maps =\
        model_detector.predict_on_batch([ftr_maps, rois])

    detector_labels = np.squeeze(detector_label_maps, axis=0)
    detector_deltas = np.squeeze(detector_delta_maps, axis=0)
    detector_deltas = np.reshape(detector_deltas, (detector_deltas.shape[0],-1, 4))

    maxIndexes = np.argmax(detector_labels, axis=1)
    bg_mask = maxIndexes!=0 # only want bg_score < 0.5 proposals
    frcnn_proposals = rpn_proposals[bg_mask]

    ### calculate non-bg score
    selected_labels = detector_labels[bg_mask]
    frcnn_scores = np.amax(selected_labels, axis=1)

    deltas_raws = detector_deltas[bg_mask]
    oj_indexes = np.argmax(selected_labels, axis=1)
    deltas = []
    for j,oj_index in enumerate(oj_indexes):
        deltas.append(deltas_raws[j,oj_index,:])
    deltas = np.array(deltas, dtype=np.float32)

    # print(deltas.shape)
    for proposal, score, delta in zip(frcnn_proposals, frcnn_scores, deltas):
        o1 = {'FileName':imgName,\
            'XMin':proposal[0], 'XMax':proposal[1],\
            'YMin':proposal[2], 'YMax':proposal[3],\
            'Score':score,\
            'ClassName':11}

        proposal = delta_to_bbox(delta)
        o2 = {'FileName':imgName,\
            'XMin':proposal[0], 'XMax':proposal[1],\
            'YMin':proposal[2], 'YMax':proposal[3],\
            'Score':score,\
            'ClassName':11}

        df_o1 = df_o1.append(o1, ignore_index=True)
        df_o2 = df_o2.append(o2, ignore_index=True)

pred_f0 = C.sub_data_dir.joinpath('mc_bbox_rpn_prediction_validation.csv')
pred_f1 = C.sub_data_dir.joinpath('mc_bbox_cls_prediction_validation.csv')
pred_f2 = C.sub_data_dir.joinpath('mc_bbox_prediction_validation.csv')
df_o0.to_csv(pred_f0)
df_o1.to_csv(pred_f1)
df_o2.to_csv(pred_f2)

C.set_validation_prediction(pred_f1)

pickle_path = Path.cwd().joinpath('frcnn.train.config.pickle')
pickle.dump(C, open(pickle_path, 'wb'))
