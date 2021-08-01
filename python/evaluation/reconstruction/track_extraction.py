import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt

import PIL
from PIL import Image

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Abstract import binning_objects
from Information import *
from Configuration import extractor_config
from Layers import *
from Architectures import FC_DenseNet
from HitGenerators import Stochastic


from track_detection import define_nms_fn
from track_detection import track_detection


### reconstruct FC-DenseNet
input_shape = (256, 256, 1)
architecture = FC_DenseNet(input_shape, 3, dr=0.1)
extractor_model = architecture.get_model()
weight_file_name = 'photographic_mc_arc_FCDense_Dropout_0.1_Unaugmented.h5'
extractor_model.load_weights(str(Path.cwd().joinpath(weight_file_name)), by_name=True)

### make some convenient functions
def select_hits_in_bbox(hits, bbox):
    x_bins = [-810, bbox[0]-1, bbox[1]+1, 810]
    y_bins = [-810, bbox[2]-1, bbox[3]+1, 810]
    xs, ys = [], []
    for id, pos in hits.items():
        xs.append(pos[0])
        ys.append(pos[1])
    hit_selected_by_x = binning_objects(hits, xs, x_bins)[2]
    hit_selected_by_y = binning_objects(hits, ys, y_bins)[2]
    selected_ids = set(hit_selected_by_x).intersection(set(hit_selected_by_y))
    selected_hits = {id:hits[id] for id in selected_ids}
    return selected_hits

def make_density_photo(hits, resolution=256):
    selected_mcs_x, selected_mcs_y = [], []
    for key, pos in hits.items():
        selected_mcs_x.append(pos[0])
        selected_mcs_y.append(pos[1])

    sorted_selected_mcs_x = deepcopy(selected_mcs_x)
    sorted_selected_mcs_x.sort()

    sorted_selected_mcs_y = deepcopy(selected_mcs_y)
    sorted_selected_mcs_y.sort()

    # create the blank input photo by resolution and the xy ratio
    xmin = sorted_selected_mcs_x[0]
    xmax = sorted_selected_mcs_x[-1]
    ymin = sorted_selected_mcs_y[0]
    ymax = sorted_selected_mcs_y[-1]
    x_delta = xmax - xmin
    y_delta = ymax - ymin
    ratio = y_delta/x_delta

    xpixel, ypixel = resolution, resolution
    density_photo = np.zeros(shape=(ypixel,xpixel), dtype=np.uint8)
    id_photo = [ [ [] for _ in range(xpixel) ] for _ in range(ypixel)]

    # setup the x and y grids that are for sorting particles
    xstep = x_delta/xpixel
    ystep = y_delta/ypixel
    xbins = [xmin + i*xstep for i in range(xpixel+1)]
    xbins[-1] = xbins[-1]+1
    ybins = [ymin + i*ystep for i in range(ypixel+1)]
    ybins[-1] = ybins[-1]+1

    # fill the density in the blank photo and truth
    bins_by_row = binning_objects(hits, selected_mcs_y, ybins)[1:]
    for row, bin in enumerate(bins_by_row):
        x_bin_flatten = [ hits[id][0] for id in bin]
        squares_by_column = binning_objects(bin, x_bin_flatten, xbins)[1:]
        for col, square in enumerate(squares_by_column):
            density = len(square)#number density
            density_photo[ypixel-row-1][col] = density
            if density!=0:
                for id in square:
                    id_photo[ypixel-row-1][col].append(id)

    # convert density photo from uint8 to float32
    im_in = Image.fromarray(density_photo)
    density_photo = np.array(im_in, dtype=np.float32)

    return density_photo, id_photo

### define a process-oriented function
def track_extraction(hit_dict, bboxes):
    # The most complete hit ids
    # Some ids will be removed if they are predicted to form a track
    hit_ids = list(hit_dict.keys())
    tracks = []

    # loop over bboxes; the order of bboxes was decided by the
    # object detection scores.
    for bbox in bboxes:
        track = []

        # select the remainding hits
        hits_s = {id:hit_dict[id] for id in hit_ids}# s=selected
        # select hits in the bounding box
        hits_s = select_hits_in_bbox(hits_s, bbox)

        # if no hits in the bounding box, skip to the next iteration
        if len(hits_s) <3:
            continue
        # otherwise, grouping hits into tracks
        else:
            density_photo, id_photo=make_density_photo(hits_s)

        # predict the probality that a cell contains a track hit
        density_photo = np.expand_dims(density_photo, axis=0)
        scores = extractor_model.predict_on_batch(density_photo)
        scores = np.squeeze(scores, axis=0)

        # select hits by score
        maxIndices = np.argmax(scores, axis=2)
        masks = maxIndices==2

        # move hits to a track list
        for i, row in enumerate(masks):
            for j, mask in enumerate(row):
                if mask==True:
                    cell = id_photo[i][j]
                    if len(cell)!=0:
                        for id in cell:
                            track.append(id)
                            hit_ids.remove(id)

        # append this track to the summary of tracks
        tracks.append(track)

    return tracks





### test bench
if __name__ == "__main__":
    track_dir = Path('/home/Billy/Mu2e/analysis/DLTracking/tracks')
    dp_list =  ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
            "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art"]
    db_list = [ track_dir.joinpath(dp+'.db') for dp in dp_list]

    mean = 5.0
    std = 2.0
    dist = norm(loc=mean, scale=1/std)

    gen = Stochastic(dist=dist, db_files=db_list, hitNumCut=20)
    windowNum = 10

    nms = define_nms_fn(max_output_size=3000,\
            iou_threshold=.7, score_threshold=.5, soft_nms_sigma=1400.0)

    for i in range(windowNum):
        sys.stdout.write(t_info(f'Processing window: {i+1}/{windowNum}', '\r'))
        if i+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        hit_dict, trks_ref = gen.generate(mode='eval')
        bboxes = track_detection(hit_dict, nms)
        trks_pred = track_extraction(hit_dict, bboxes)
