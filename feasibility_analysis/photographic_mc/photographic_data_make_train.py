# feasibility test
# detect outliner by CNN

import sys
import csv
from pathlib import Path
from collections import Counter
import timeit
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from abstract import binning_objects
from TrackDB_Classes import *
from mu2e_output import *


def make_data_from_distribution(track_dir, mean, std, windowNum):

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 9

    ### Construct Path Objects
    dp_list = ["dig.mu2e.CeEndpoint.MDC2018b.001002_00000011.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000012.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000014.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000020.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000024.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000044.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000169.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000172.art",\
                "dig.mu2e.CeEndpoint.MDC2018b.001002_00000192.art"]
    dp_name_iter = iter(dp_list)
    dp_name = next(dp_name_iter)
    db_file = track_dir.joinpath(dp_name+".db")
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    photographic_train_dir = data_dir.joinpath('photographic_train')
    photographic_train_dir.mkdir(parents=True, exist_ok=True)
    input_file = photographic_train_dir.joinpath('inputs.npy')
    output_file = photographic_train_dir.joinpath('outputs.npy')

    pinfo('Importing the photographic grid')
    ### pixel truth labels
    is_blank = np.array([1,0,0], dtype=bool)
    is_bg = np.array([0,1,0], dtype=bool)
    is_major = np.array([0,0,1], dtype=bool)

    ### Construct the photographic grid
    blank_photo = np.zeros(shape=(800,800),dtype=np.int8)
    step = 1620/800
    xbins = [ -810+i*step for i in range(801) ]
    ybins = deepcopy(xbins)

    blank_truth = np.zeros(shape=(800,800,3),dtype=bool)
    blank_truth[:,:,0]=True

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
    inputs = []
    outputs = []
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

            strawHit_qrl = session.query(StrawHit).filter(StrawHit.particle==ptcl.id)
            hitNum = strawHit_qrl.count()
            if hitNum < hitNumCut:
                continue
            else:
                track_box.append(ptcl)
                track_found_number += 1

        # draw bbox for each track
        mcs = [ session.query(StrawDigiMC).filter(StrawDigiMC.particle==ptcl.id).all() for ptcl in track_box ]

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
            x_bins = [-810, bbox[0], bbox[1], 810]
            y_bins = [-810, bbox[2], bbox[3], 810]

            # get position tuples in the bounding box
            pos_selected_by_x = binning_objects(mcs_pos_flatten, xs_flatten, x_bins)[2]
            pos_selected_by_y = binning_objects(mcs_pos_flatten, ys_flatten, y_bins)[2]
            selected_mcs_pos = list(set(pos_selected_by_x).intersection(pos_selected_by_y))
            selected_mcs_y = [ y for [x,y,z] in selected_mcs_pos ]

            ### fill the density in the blank photo and truth
            input_photo = np.zeros(shape=(800,800),dtype=np.int8)
            output_truth =  np.zeros(shape=(800,800,3),dtype=bool)
            output_truth[:,:,0] = True

            # first index is row
            bins_by_row = binning_objects(selected_mcs_pos, selected_mcs_y, ybins)[1:]
            if len(bins_by_row)!=800:
                perr(f'Number of bins by row is not equal to 800, value is{len(bins_by_row)}')
                sys.exit()

            for row, bin in enumerate(bins_by_row):
                x_bin_flatten = [ x for (x,y,z) in bin]
                squares_by_column = binning_objects(bin, x_bin_flatten, xbins)[1:]
                for col, square in enumerate(squares_by_column):
                    density = len(square)#number density
                    input_photo[799-row][col] = density
                    if density != 0 :
                        has_major = False
                        for pos in square:
                            if pos in mcs_pos[i]:
                                has_major = True
                                break
                        if has_major == True:
                            output_truth[799-row][col] = is_major
                        else:
                            output_truth[799-row][col] = is_bg

            if len(np.where(input_photo!=0)[0]) == 0:
                pdebug(bins_by_row)
                ptcl_found = False
                for row, bin in enumerate(bins_by_row):
                    x_bin_flatten = [ x for (x,y,z) in bin]
                    squares_by_column = binning_objects(bin, x_bin_flatten, xbins)[1:]
                    for square in squares_by_column:
                        if len(square)!= 0:
                            ptcl_found = True
                            break
                    if ptcl_found:
                        pdebug(squares_by_column,f'{799-row}th row')
                    elif len(bin)!=0 :
                        pdebug(bin,'bin')
                        pdebug(x_bin_flatten, 'xbin_flatten')
                        pdebug(xbins, 'xbins')
                        pdebug(binning_objects(bin, x_bin_flatten, xbins))
                pdebug('Empty input photo!')
                sys.exit()
            inputs.append(input_photo)
            outputs.append(output_truth)

    inputs = np.array(inputs,dtype=np.int8)
    outputs = np.array(outputs,dtype=bool)

    np.save(input_file, inputs)
    np.save(output_file, outputs)

    return photographic_train_dir, input_file, output_file


def make_data(C, mode):
    pstage("Making training data")


    if mode == "normal":
        track_dir = C.track_dir
        mean = C.trackNum_mean
        std = C.trackNum_std
        windowNum = C.window
        train_dir, X_file, Y_file = make_data_from_distribution(track_dir, mean, std, windowNum)

    C.set_inputs(train_dir, X_file, Y_file)
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('photographic.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
    return C

if __name__ == "__main__":
    pbanner()
    psystem('Photographic track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    track_str = '../../tracks'
    track_dir = Path(track_str)
    C = Config(track_dir)

    mode = 'normal'
    window = 200 # unit: number of windows
    mean = 10
    std = 3

    track_dir = Path(track_str)
    C = Config(track_dir)

    C.set_distribution(mean, std)
    C.set_window(window)

    start = timeit.default_timer()
    make_data(C, mode)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')
