"""
This script prepares input photos and output truths without random masking.
"""
# feasibility test
# detect outliner by CNN

import sys
import csv
from pathlib import Path
import shutil
from collections import Counter
import timeit
import pickle
from copy import deepcopy

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import *

import PIL
from PIL import Image

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config
from Abstract import binning_objects
from Database import *
from Information import *

def make_data_from_distribution(C):

    track_dir = C.track_dir
    mean = C.trackNum_mean
    std = C.trackNum_std
    windowNum = C.window
    resolution = C.resolution

    dRatios = []

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 20

    ### Construct Path Objects
    dp_list = C.train_dp_list
    dp_name_iter = iter(dp_list)
    dp_name = next(dp_name_iter)
    db_file = track_dir.joinpath(dp_name+".db")

    ### pixel truth labels
    is_blank = np.array([1,0,0], dtype=np.float32)
    is_bg = np.array([0,1,0], dtype=np.float32)
    is_major = np.array([0,0,1], dtype=np.float32)

    ### initialize sqlite session
    # Connect to the database
    pinfo('Connecting to the track database')
    engine = create_engine('sqlite:///'+str(db_file))
    # make session
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine) # session factory
    session = Session() # session object

    ### make bins
    pinfo('Making tracking windows')

    # get a distribution of integers
    floats = np.random.normal(loc=mean, scale=std, size=windowNum)
    float_type_ints = np.around(floats)
    track_numbers = float_type_ints.astype(int)

    # get particles and the particle iterator
    ptcls = session.query(Particle).all()
    ptcl_iter = iter(ptcls)

    # get major tracks for each
    index=0
    for idx, track_number in enumerate(track_numbers):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
        if idx+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # get corresponding number of tracks in this window
        track_box = []
        track_found_number = 0
        while track_found_number < track_number:
            try:
                ptcl = next(ptcl_iter)
            except:
                pinfo('\nRun out of particles')
                dp_name = next(dp_name_iter)
                db_file = track_dir.joinpath(dp_name+".db")
                pinfo('Connecting to the next track database')
                engine = create_engine('sqlite:///'+str(db_file))
                Session = sessionmaker(bind=engine) # session factory
                session = Session() # session object
                ptcls = session.query(Particle).all()
                ptcl_iter = iter(ptcls)
                ptcl = next(ptcl_iter)
                track_box = [ptcl]
                track_found_number = 1

            strawHit_qrl = session.query(StrawDigiMC).filter(StrawDigiMC.particle==ptcl.id)
            hitNum = strawHit_qrl.count()
            if hitNum < hitNumCut:
                continue
            else:
                track_box.append(ptcl)
                track_found_number += 1

        # draw bbox for each track
        mcs = [ session.query(StrawDigiMC).filter(StrawDigiMC.particle==ptcl.id).all() for ptcl in track_box ]

        for mc in mcs:
            if len(mc) < hitNumCut:
                pdebug(f'Less than {hitNumCut}')
        # xs, ys, zs are 2-D lists.
        # Every list inside corresponds to a particle
        # They are going to be the reference for drawing bounding boxes
        xs = [ [mc.x for mc in mcs_for_ptcl] for mcs_for_ptcl in mcs ]
        ys = [ [mc.y for mc in mcs_for_ptcl] for mcs_for_ptcl in mcs ]
        zs = [ [mc.z for mc in mcs_for_ptcl] for mcs_for_ptcl in mcs ]

        # flatten data means destroying the 2-D list structure so that you cannot
        # tell which (x,y,z) belong to which particle.
        # They will be collections of data of all particles in this window
        mcs_pos = [ [(x,y,z) for x, y, z in zip(xs_i, ys_i, zs_i)] for xs_i, ys_i, zs_i in zip(xs, ys, zs) ]
        mcs_pos_flatten = [ (x,y,z) for mcs_pos_i in mcs_pos for x,y,z in mcs_pos_i ]
        xs_flatten = [ x for x, y, z in mcs_pos_flatten]
        ys_flatten = [ y for x, y, z in mcs_pos_flatten]

        bboxes = [ [min(xs_i), max(xs_i), min(ys_i), max(ys_i)] for xs_i, ys_i in zip(xs, ys) ]

        for i, bbox in enumerate(bboxes):
            # make x bins and y bins for binning objects
            x_bins = [-810, bbox[0]-1, bbox[1]+1, 810]
            y_bins = [-810, bbox[2]-1, bbox[3]+1, 810]

            # get position tuples in the bounding box
            pos_selected_by_x = binning_objects(mcs_pos_flatten, xs_flatten, x_bins)[2]
            pos_selected_by_y = binning_objects(mcs_pos_flatten, ys_flatten, y_bins)[2]
            selected_mcs_pos = list(set(pos_selected_by_x).intersection(pos_selected_by_y))
            selected_mcs_x = [ x for [x,y,z] in selected_mcs_pos ]
            sorted_selected_mcs_x = deepcopy(selected_mcs_x)
            sorted_selected_mcs_x.sort()
            selected_mcs_y = [ y for [x,y,z] in selected_mcs_pos ]
            sorted_selected_mcs_y = deepcopy(selected_mcs_y)
            sorted_selected_mcs_y.sort()

            # create the blank input photo by resolution and the xy ratio
            xmin = sorted_selected_mcs_x[0]
            xmax = sorted_selected_mcs_x[-1]
            ymin = sorted_selected_mcs_y[0]
            ymax = sorted_selected_mcs_y[-1]
            x_delta = xmax - xmin
            y_delta = ymax - ymin

            for i in range(len(sorted_selected_mcs_x)-1):
                dx = sorted_selected_mcs_x[i+1] - sorted_selected_mcs_x[i]
                dRatio = dx/x_delta
                dRatios.append(dRatio)

            for i in range(len(sorted_selected_mcs_y)-1):
                dy = sorted_selected_mcs_y[i+1] - sorted_selected_mcs_y[i]
                dRatio = dx/y_delta
                dRatios.append(dRatio)


    dRatios = [x*256 for x in dRatios]
    plt.hist(dRatios, 20)
    plt.xlabel('Pixel')
    plt.ylabel('Number of Hits')
    plt.title('Nearest Hits\' 1D Pixel Distances')
    plt.show()

    return


def make_data(C, mode):
    """
    This function helps determine which mode should be used when preparing training data.
    If mode is "normal", track number would fit a Gaussian distribution whose parameters were specificed in the configuration object before called.
    If mode is " " [Unfinished]
    """
    pstage("Making training data")


    if mode == "normal":
        make_data_from_distribution(C)
    return C

if __name__ == "__main__":
    pbanner()
    psystem('Photographic track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    track_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/tracks'
    data_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/data'

    track_dir = Path(track_dir_str)
    data_dir = Path(data_dir_str)

    C = extractor_config(track_dir, data_dir)

    mode = 'normal'
    window = 1000 # unit: number of windows
    mean = 5
    std = 2
    resolution = 256

    dp_list = ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000011.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000012.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000014.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000020.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000024.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000044.art"]

    C.set_train_dp_list(dp_list)

    C.set_distribution(mean, std)
    C.set_window(window)
    C.set_resolution(resolution)

    start = timeit.default_timer()
    make_data(C, mode)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')
