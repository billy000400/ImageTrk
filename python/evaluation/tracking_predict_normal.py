# import begins
import sys
from pathlib import Path
from math import sqrt
import timeit
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import(
    Input,
    Conv2D,
    Dense,
    MaxPool2D,
    Dropout,
    Flatten
)
from tensorflow.keras.preprocessing import sequence
from tensorflow.image import non_max_suppression_with_scores

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_rpn import rpn
from frcnn_util import(
    binning_objects,
    make_anchors,
    propose_score_bbox_list
)

from base_net import vgg16
from TrackDB_Classes import *
from mu2e_output import *
### import ends




pbanner()
psystem('Tracking with Faster R-CNN and a CNN Extractor')
pmode('Evaluating')





### setup parameters
hitsNumCut = 9
windowNum = 400
mean = 10
std = 3
np.random.seed(0)

track_str = '/home/Billy/Mu2e/analysis/data/tracks'
db_name = 'dig.mu2e.CeEndpoint.MDC2018b.001002_00000172.art'
input_shape = (800, 800, 3)
anchor_scales = [150,250,350]
anchor_ratios = [[1,1],\
                    [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]
ratio = 16 # vgg16

### setup io
db_file = Path(track_str).joinpath(db_name+'.db')
weights_dir = Path.cwd().parent.parent.joinpath('weights')
frcnn_weights = weights_dir.joinpath('rpn_00.h5')
extractor_weights = weights_dir.joinpath('extractor_00.h5')
tracks = []
data_dir = Path.cwd().parent.parent.joinpath('data')
prediction_file = data_dir.joinpath('tracks_prediction.csv')

### Make tracking windows
pinfo("Start Making Tracking Window")
floats = np.random.normal(loc=mean, scale=std, size=windowNum)
float_type_ints = np.around(floats)
track_numbers = float_type_ints.astype(int)

### prepare particles
pinfo('Connecting to the track database')
engine = create_engine('sqlite:///'+str(db_file))
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()
ptcls = session.query(Particle).all()
ptcl_iter = iter(ptcls)

### make groups
position_groups = []
id_groups = []
times = []
groupNum = windowNum
for idx, track_number in enumerate(track_numbers):
    sys.stdout.write(t_info(f'Loading Reconstructed StrawHits for window {idx+1}/{groupNum}', special='\r'))
    if idx+1 == groupNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    found = 0
    positions = []
    ids = []
    while found < track_number:
        ptcl = next(ptcl_iter)
        hit_qrl = session.query(StrawHit).filter(StrawHit.particle==ptcl.id)
        if hit_qrl.count() < hitsNumCut:
            continue
        else:
            hits = hit_qrl.all()
            for hit in hits:
                ids.append(hit.id)
                positions.append((hit.x_reco, hit.y_reco, hit.z_reco))
                times.append(hit.t_reco)
            found += 1
    position_groups.append(positions)
    id_groups.append(ids)

### obtain info from groups
t_min = min(times)
t_max = max(times)
hitsNum = len(times)

### Make anchors
anchors = make_anchors(input_shape, ratio, anchor_scales, anchor_ratios)

### prepare FRCNN
pinfo('Assembling the Faster R-CNN')
pinfo('A test run is starting')
rpn = rpn(anchor_scales=anchor_scales, anchor_ratios=anchor_ratios)
frcnn_input_layer = Input(shape=input_shape)
base_net = vgg16()
x = base_net.nn(frcnn_input_layer)
classifier = rpn.classifier(x)
regressor = rpn.regression(x)
frcnn = Model(inputs=frcnn_input_layer, outputs = [classifier,regressor])
frcnn.load_weights(frcnn_weights, by_name=True)
frcnn.predict(x=np.zeros((1, input_shape[0], input_shape[1], input_shape[2])), batch_size=1)
pinfo('Finished the test run')

### prepare extractor
pinfo('Assembling the CNN track extractor')
extractor_input_layer = Input(shape=(1000, 3, 1))
x = Conv2D(32, (10, 3), padding='same', activation='relu')(extractor_input_layer)
x = Dropout(0.2)(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
extractor_output_layer = Dense(1000, activation='sigmoid')(x)
extractor = Model(inputs=extractor_input_layer, outputs=extractor_output_layer)
extractor.load_weights(extractor_weights, by_name=True)


####### Start Timing #######
pinfo("Timer starts!")
start = timeit.default_timer()

### Parse every tracking window
groupNum = len(position_groups)
for idx, pos_group in enumerate(position_groups):
    sys.stdout.write(t_info(f'Parsing Windows {idx+1}/{groupNum}', special='\r'))
    if idx+1 == groupNum:
        sys.stdout.write('\n')
    sys.stdout.flush()

    # escape empty window
    if len(pos_group) == 0: continue

    # get x y z
    xs = []
    ys = []
    zs = []
    for pos in pos_group:
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

    # make the image input(in buffer)
    fig, ax = plt.subplots(figsize=(8,8), dpi=800/8, frameon=False)
    ax.scatter(xs, ys, c='b', s=0.2, alpha=0.3)
    ax.axis('scaled')
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img = np.fromstring(buf, dtype=np.uint8).reshape(1, 800, 800, 3)
    plt.close()

    # frcnn predicting with soft nms
    score_maps, delta_maps = frcnn.predict(x=img, batch_size=1)
    score_map = score_maps[0]
    delta_map = delta_maps[0]
    score_bbox_raw = propose_score_bbox_list(anchors, input_shape, score_map, delta_map)

    scores = []
    bboxes_raw = []
    for score, bbox in score_bbox_raw:
        scores.append(score)
        bboxes_raw.append(bbox)

    scores = tf.constant(scores, dtype=tf.float32)
    bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
    bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)

    selected_indices, selected_score =\
        non_max_suppression_with_scores(bboxes_raw_tf, scores,\
                max_output_size=20,\
                iou_threshold=1.0, score_threshold=0.9,\
                soft_nms_sigma=0.0)

    if len(selected_indices) == 0:
        pwarn("No bbox proposed")

    selected_indices_list = selected_indices.numpy().tolist()
    bboxes_in_img = [ bboxes_raw[index] for index in selected_indices_list ]

    # trimming bboxes
    for i, bbox in enumerate(bboxes_in_img):
        if bbox[0] < 0:
            bboxes_in_img[i][0] = 0
        if bbox[1] > 1:
            bboxes_in_img[i][1] = 1
        if bbox[2] < 0:
            bboxes_in_img[i][2] = 0
        if bbox[3] > 1:
            bboxes_in_img[i][3] = 1
        if bbox[1] < bbox[0]:
            pwarn(f"bbox {i} XMax is less than XMin")
        if bbox[3] < bbox[2]:
            pwarn(f"bbox {i} YMax is less than YMin")

    # translate normalized bboxes to unnormalized bboxes
    bboxes_in_img = [ [ (el-0.5)*1620 for el in bbox] for bbox in bboxes_in_img ]

    # grouping hits into bboxes
    hit_boxes_raw = []
    for bbox in bboxes_in_img:
        xbins = [-810, bbox[0], bbox[1], 810]
        ybins = [-810, bbox[2], bbox[3], 810]

        # pdebug(pos_group)
        # pdebug(ys)


        pos_selected_by_x = binning_objects(pos_group, xs, xbins)[2]
        if len(pos_selected_by_x) == 0:
            continue

        pos_selected_by_y = binning_objects(pos_group, ys, ybins)[2]
        if len(pos_selected_by_y) == 0:
            continue

        selected_hits_pos = list(set(pos_selected_by_x).intersection(pos_selected_by_y))
        selected_hits_z = [ z for x,y,z in selected_hits_pos ]


        pos_z_raw  = dict(zip(selected_hits_pos, selected_hits_z))
        pos_z = sorted( pos_z_raw.items(), key=lambda item: item[1])

        hit_box = [coordinate for coordinate in pos for pos,z in pos_z]
        hit_boxes_raw.append(hit_box)

    # padding hits for every hit_box
    hit_boxes = []
    for row in hit_boxes_raw:
        if len(row) > 3000:
            row_last = row
            while len(row_last) > 3000:
                hit_boxes.append(row_last[:3000])
                row_last = row_last[3000:]
        else:
            hit_boxes.append(row)
    hit_boxes = sequence.pad_sequences(hit_boxes, maxlen=3000,dtype='float32',padding='post',value=0.0)
    hit_boxes = np.reshape(hit_boxes, (len(hit_boxes), -1, 3))

    # cnn extractor predicting
    hits_truths = extractor.predict(x=hit_boxes, batch_size=1)
    hits_truths = hits_truths.tolist()

    # extract tracks
    for hits_truth in hits_truths:
        track = []
        for i, hit_truth in enumerate(hits_truth):
            if hit_truth >= 0.5:
                try:
                    track.append(id_groups[idx][i])
                except:
                    continue
        if len(track)==0:
            continue
        else:
            tracks.append(track)

### output time info
total_time = timeit.default_timer()-start
print('\n')
pinfo(f'Elapsed time: {total_time}(sec)')
pinfo(f"Time in experiment: {t_max-t_min}(ns)")
pinfo(f"Number of hits: {hitsNum}")


### output tracks to local
prediction_file_open = open(prediction_file, 'w')
writer = csv.writer(prediction_file_open)
for track in tracks:
    writer.writerow(track)
prediction_file_open.close()
