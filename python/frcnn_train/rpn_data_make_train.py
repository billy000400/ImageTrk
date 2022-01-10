# @Author: Billy Li <billyli>
# @Date:   01-05-2022
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 01-05-2022



# Make Training Set
# Author: Billy Haoyang Li

# General import

"""@package docstring
Documentation for this module.

More details.
"""

import sys
from pathlib import Path
import shutil
import timeit
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from HitGenerators import Event_V2 as Event
from Abstract import zt2map
from Configuration import frcnn_config
from Information import *






def make_data_from_generator(C):

    ### prepare parameters
    windowNum = C.window
    resolution = C.resolution
    hitNumCut = 10

    ### prepare event generators
    dp_list = C.train_dp_list
    gen = Event(dp_list, hitNumCut)

    ### construct Path objects
    data_dir = C.data_dir
    C.sub_data_dir = data_dir.joinpath(Path.cwd().name)
    data_dir = C.sub_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    weight_dir = Path.cwd()
    weight_path = weight_dir.joinpath('wcnn_00.h5')

    img_dir = data_dir.joinpath('imgs_train')
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "bbox_proposal_train.csv"
    bbox_file = data_dir.joinpath(csv_name)

    ### build wcnn for window proposals
    wcnn = Img2Vec(input_shape=(256, 256, 1)).get_model()
    wcnn.load_weights(\
            str(cwd.joinpath(weight_path)), by_name=True)


    # get tracks for each event
    bbox_table_row_num = 0
    dict_for_df = {}
    for idx in range(windowNum):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
        if idx+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # initialize parameters
        img_name = str(idx+1).zfill(5)+'.png'
        img_file = img_dir.joinpath(img_name)

        x_all = []
        y_all = []

        rects = []

        # generate an event
        hit_all, track_all = gen.generate(mode='eval')

        # filter out long staying particle
        hitsInTracks = []
        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[:-1]
            hitsPerTrack = [hit_all[idx] for idx in hitIdc]
            tsPerTrack = [hit[3] for hit in hitsPerTrack]
            delta_t= max(tsPerTrack)-min(tsPerTrack)
            if delta_t > 1000:
                continue
            hitsInTracks.append(hitsPerTrack)

        # make zt maps to propose tracking windows
        hits = [hit for hitsPerTrack in hitsInTracks for hit in hitsPerTrack]
        zs = [hit[2] for hit in hits]
        ts = [hit[3] for hit in hits]
        map = zt2map(zs, ts, resolution)

        # wcnn proposing tracking windows
        vec1, vec2 = wcnn.predict_on_batch(map)
        windows = delta_to_roi_1D(vec1, vec2)

            xs = [hit.x for hit in strawHits]
            ys = [hit.y for hit in strawHits]
            x_all = x_all + xs
            y_all = y_all + ys

            xs = np.array(xs)
            ys = np.array(ys)

            XMin = xs.min()
            XMax = xs.max()
            YMin = ys.min()
            YMax = ys.max()

            rect = Rectangle((XMin,YMin), XMax-XMin, YMax-YMin,linewidth=1,edgecolor='r',facecolor='none')
            rects.append(rect)

            xmin = XMin/1620 + 0.5 -0.01
            xmax = XMax/1620 + 0.5 + 0.01
            ymin = YMin/1620 + 0.5 -0.01
            ymax = YMax/1620 + 0.5 +0.01
            pdgId = session.query(Particle.pdgId).filter(Particle.id==ptcl.id).one_or_none()[0]
            dict_for_df[bbox_table_row_num] = {'FileName':img_name,\
                                    'XMin':xmin,\
                                    'XMax':xmax,\
                                    'YMin':ymin,\
                                    'YMax':ymax,\
                                    'ClassName':pdgId}

            bbox_table_row_num += 1
            track_found_num += 1

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        layout = {'pad':0, 'h_pad':0, 'w_pad':0, 'rect':(0,0,1,1) }
        plt.figure(figsize=(8,8), dpi=resolution/8, frameon=False, tight_layout=layout)
        plt.scatter(x_all, y_all, c='b', s=1)
        plt.axis('scaled')
        plt.axis('off')
        plt.xlim([-810, 810])
        plt.ylim([-810, 810])
        # for rect in rects:
        #     plt.gca().add_patch(rect)
        # plt.show()
        plt.savefig(img_file, bbox_inches='tight', pad_inches=0)
        plt.close()

    train_df = pd.DataFrame.from_dict(dict_for_df, "index")
    train_df.to_csv(bbox_file)

    return bbox_file, img_dir

def make_data(C):

    pstage('Make Raw Data')


    bbox_file, img_dir = make_data_from_generator(C)
    C.set_raw_training_data(bbox_file, img_dir)

    pcheck_point('Images and the bbox table')
    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    # initialize parameters
    track_dir_str = '../../tracks'
    data_dir_str = '../../data'

    window = 700 # unit: number of windows
    resolution = 512

    dp_list =  ["train"]

    # setup parameters
    track_dir = Path(track_dir_str)
    data_dir = Path(data_dir_str)
    C = frcnn_config(track_dir, data_dir)
    C.set_train_dp_list(dp_list)
    C.set_window(window)
    C.set_resolution(resolution)

    # prepare raw tarining set with time measurement
    start = timeit.default_timer()
    C = make_data(C)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
