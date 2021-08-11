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

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 20

    ### Construct Path Objects
    dp_list = C.train_dp_list
    dp_name_iter = iter(dp_list)
    dp_name = next(dp_name_iter)
    db_file = track_dir.joinpath(dp_name+".db")

    data_dir = C.data_dir
    C.sub_data_dir = data_dir.joinpath(Path.cwd().name)
    data_dir = C.sub_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    ### directories for numpy data
    photographic_train_x_dir = data_dir.joinpath('photographic_large_train_X')
    photographic_train_y_dir = data_dir.joinpath('photographic_large_train_Y')

    ### directories for jpg data
    photo_train_in_dir = data_dir.joinpath('photographic_large_train_input_photo')
    photo_train_out_dir = data_dir.joinpath('photographic_large_train_output_truth')

    shutil.rmtree(photographic_train_x_dir, ignore_errors=True)
    shutil.rmtree(photographic_train_y_dir, ignore_errors=True)
    shutil.rmtree(photo_train_in_dir, ignore_errors=True)
    shutil.rmtree(photo_train_out_dir, ignore_errors=True)
    photographic_train_x_dir.mkdir(parents=True, exist_ok=True)
    photographic_train_y_dir.mkdir(parents=True, exist_ok=True)
    photo_train_in_dir.mkdir(parents=True, exist_ok=True)
    photo_train_out_dir.mkdir(parents=True, exist_ok=True)

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
            ratio = y_delta/x_delta
            if ratio >= 1:
                xpixel = int(np.ceil(resolution/ratio))
                ypixel = resolution
                input_photo = np.zeros(shape=(ypixel,xpixel), dtype=np.uint8)
                output_truth = np.zeros(shape=(ypixel,xpixel,3), dtype=np.uint8)
                output_truth[:,:,0] = 1
            else:
                xpixel = resolution
                ypixel = int(np.ceil(resolution*ratio))
                input_photo = np.zeros(shape=(ypixel,xpixel), dtype=np.uint8)
                output_truth = np.zeros(shape=(ypixel,xpixel,3), dtype=np.uint8)
                output_truth[:,:,0] = 1

            # setup the x and y grids that are for sorting particles
            xstep = x_delta/xpixel
            ystep = y_delta/ypixel
            xbins = [xmin + i*xstep for i in range(xpixel+1)]
            xbins[-1] = xbins[-1]+1
            ybins = [ymin + i*ystep for i in range(ypixel+1)]
            ybins[-1] = ybins[-1]+1

            ### fill the density in the blank photo and truth
            # first index is row
            bins_by_row = binning_objects(selected_mcs_pos, selected_mcs_y, ybins)[1:]

            majorDetected = 0
            for row, bin in enumerate(bins_by_row):
                x_bin_flatten = [ x for (x,y,z) in bin]
                squares_by_column = binning_objects(bin, x_bin_flatten, xbins)[1:]
                for col, square in enumerate(squares_by_column):
                    density = len(square)#number density
                    input_photo[ypixel-row-1][col] = density
                    if density != 0 :
                        has_major = False
                        for pos in square:
                            if pos in mcs_pos[i]:
                                has_major = True
                                majorDetected+=1
                                break
                        if has_major == True:
                            output_truth[ypixel-row-1][col] = is_major

                        else:
                            output_truth[ypixel-row-1][col] = is_bg


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




            # if index == 240:
            #     pdebug(np.where((output_truth==is_major).all(axis=2)))


            ### scale tensors

            # calculate scaled dimensions
            res = C.resolution
            scale_want = (res, res) # notice that this is the scale for IMAGES!

            xo = input_photo
            yo = output_truth
            ratio = xo.shape[0]/xo.shape[1]
            im_in = Image.fromarray(xo)
            im_out = Image.fromarray(yo, mode='RGB')

            for degree in [0]:

                input_file = photographic_train_x_dir.joinpath(f'input_{str(index).zfill(6)}')
                output_file = photographic_train_y_dir.joinpath(f'output_{str(index).zfill(6)}')
                input_photo_file = photo_train_in_dir.joinpath(f'input_{str(index).zfill(6)}.jpg')
                output_truth_file = photo_train_out_dir.joinpath(f'output_{str(index).zfill(6)}.jpg')

                x = im_in.resize(scale_want)
                y = im_out.resize(scale_want)

                x = x.rotate(degree)
                y = y.rotate(degree)

                x = np.array(x, dtype=np.float32)
                y = np.array(y, dtype=np.float32)

                np.save(input_file, x)
                np.save(output_file, y)

                x_max = int(x.max())
                ratio = 255/x_max
                for n in range(x_max+1):
                    x[x==n] = np.uint8(255-n*ratio)

                y[(y==[1,0,0]).all(axis=2)] = np.array([255,255,255], dtype=np.float32)
                y[(y==[0,1,0]).all(axis=2)] = np.array([0,0,255], dtype=np.float32)
                y[(y==[0,0,1]).all(axis=2)] = np.array([255,0,0], dtype=np.float32)

                x = np.array(x, dtype=np.uint8)
                y = np.array(y, dtype=np.uint8)

                x = Image.fromarray(x)
                y = Image.fromarray(y, mode='RGB')
                x = x.resize( [scale_want[0]*5, scale_want[1]*5] )
                y = y.resize( [scale_want[0]*5, scale_want[1]*5] )

                x.save(input_photo_file, format='jpeg')
                y.save(output_truth_file, format='jpeg')

                index += 1


    return photographic_train_x_dir, photographic_train_y_dir


def make_data(C, mode):
    """
    This function helps determine which mode should be used when preparing training data.
    If mode is "normal", track number would fit a Gaussian distribution whose parameters were specificed in the configuration object before called.
    If mode is " " [Unfinished]
    """
    pstage("Making training data")


    if mode == "normal":
        train_x_dir, train_y_dir = make_data_from_distribution(C)

    C.set_train_dir(train_x_dir, train_y_dir)
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('photographic.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
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
    window = 3000 # unit: number of windows
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
