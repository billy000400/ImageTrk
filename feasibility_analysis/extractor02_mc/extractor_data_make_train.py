# feasibility test
# detect outliner by CNN

import sys
import csv
from pathlib import Path
from collections import Counter
import timeit
import pickle

import numpy as np
import pandas as pd

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from extractor_config import Config
from frcnn_util import binning_objects
from TrackDB_Classes import *
from mu2e_output import *

def make_data_from_dp(track_dir, dp_name, window, mode):
    # mode check
    assert (mode in ['first', 'append']),\
        t_error('Unsupported data making mode! Mode has to be either \'first\' or \'append\'')

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 9


    ### Construct Path Objects
    db_file = track_dir.joinpath(dp_name+".db")
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    extractor_train_dir = data_dir.joinpath('extractor_train')
    extractor_train_dir.mkdir(parents=True, exist_ok=True)

    X_name = "hit_pos_train.csv"
    Y_name = "hit_truth_train.csv"
    X_file = extractor_train_dir.joinpath(X_name)
    Y_file = extractor_train_dir.joinpath(Y_name)

    # open files and initialize writers
    if mode == 'first':
        open_mode = 'w'
    else:
        open_mode = 'a'
    X_open = open(X_file, open_mode)
    Xwriter = csv.writer(X_open, delimiter=',')
    Y_open = open(Y_file, open_mode)
    Ywriter = csv.writer(Y_open, delimiter=',')

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
    hits = session.query(StrawHit).order_by(StrawHit.t_reco.asc()).all() # sort by time
    # make tracking windows
    t_min = float(hits[0].t_reco)
    t_max = float(hits[-1].t_reco)
    wdNum = np.ceil((t_max-t_min)/window)
    wds = t_min + np.arange(0,wdNum+1)*window

    # group hits in windows by their time
    # this is necessory because hits in the same window spatial info are presented on
    # the same image.
    pinfo('Grouping hits into windows')
    hits = session.query(StrawHit).all()
    hit_times = [hit.t_reco for hit in hits]
    hit_groups = binning_objects(hits, hit_times, wds)
    groupNum = len(hit_groups)

    # make hit group dictionary for reference
    hitId_groupIdx_dict = {}
    for idx, group in enumerate(hit_groups):
        for hit in group:
            hitId_groupIdx_dict[hit.id] = idx

    ### Making bbox table
    pinfo('Making the hit truth table')
    ptcl_qry = session.query(Particle.id)
    ptcl_ids = [ptcl for ptcl, in ptcl_qry]
    num_ptcl = ptcl_qry.count()
    for index, ptcl_id in enumerate(ptcl_ids):
        sys.stdout.write(t_info(f'Parsing particles {index+1}/{num_ptcl}', special='\r'))
        if index+1 == num_ptcl:
            sys.stdout.write('\n')
        sys.stdout.flush()

        strawHits_qrl = session.query(StrawHit).filter(StrawHit.particle==ptcl_id)
        hitNum = strawHits_qrl.count()
        # check num of hit belongs to the ptcl
        if hitNum < hitNumCut:
            continue
        # understand which groups hits belong
        strawHits = strawHits_qrl.all()
        duplicate_group_list = [hitId_groupIdx_dict[strawHit.id] for strawHit in strawHits]
        # understand how frequent a group is(how many hits belong to it)
        cnts = Counter(duplicate_group_list).most_common()
        # select groups which have more than hitNumCut hits for this particle
        groups = [group for group, frequency in cnts if frequency >= hitNumCut]
        for group in groups:
            # get the bbox interval(2d)
            strawHits_of_the_ptcl_in_group = [ hit for hit in strawHits if (hitId_groupIdx_dict[hit.id]==group) ]
            strawHits_pos_of_the_ptcl_in_group = [ (hit.x_reco, hit.y_reco, hit.z_reco) for hit in strawHits_of_the_ptcl_in_group]
            x_all = [ x for (x,y,z) in strawHits_pos_of_the_ptcl_in_group]
            y_all = [ y for (x,y,z) in strawHits_pos_of_the_ptcl_in_group]
            z_all = [ z for (x,y,z) in strawHits_pos_of_the_ptcl_in_group]


            x_all = np.array(x_all)
            y_all = np.array(y_all)
            XMin = x_all.min()
            XMax = x_all.max()
            YMin = y_all.min()
            YMax = y_all.max()

            # get all straw hits and their positions in this group
            strawHits_in_group = [ hit for hit in hits if (hitId_groupIdx_dict[hit.id]==group) ]
            strawHits_in_group_pos = [ (hit.x_reco,hit.y_reco,hit.z_reco) for hit in strawHits_in_group ]

            # make x_flag, y_flag for binning strawHits into bbox
            strawHits_in_group_x = [ x for (x, y, z) in strawHits_in_group_pos ]
            strawHits_in_group_y = [ y for (x, y, z) in strawHits_in_group_pos ]

            # make bins
            xbins = [-810, XMin, XMax, 810]
            ybins = [-810, YMin, YMax, 810]

            # hits in the bbox is the intersection of hits in x interval and hits in y interval
            strawHits_pos_in_x_interval = binning_objects(strawHits_in_group_pos, strawHits_in_group_x, xbins)[2]
            strawHits_pos_in_y_interval = binning_objects(strawHits_in_group_pos, strawHits_in_group_y, ybins)[2]
            strawHits_pos_in_bbox = list(set(strawHits_pos_in_x_interval).intersection(set(strawHits_pos_in_y_interval)))


            # sort the strawHits in this box by their z coordinate
            strawHits_z_in_bbox = [ z for (x, y, z) in strawHits_pos_in_bbox ]
            pos_z_raw  = dict(zip( strawHits_pos_in_bbox, strawHits_z_in_bbox))
            pos_z = sorted( pos_z_raw.items(), key=lambda item: item[1])

            # prepare pos list and truth list for X and Y files

            hits_pos = []
            hits_truth = []
            for pos, z in pos_z:
                is_major = False
                if pos in strawHits_pos_of_the_ptcl_in_group:
                    is_major = True

                for coordinate in pos:
                    hits_pos.append(coordinate)
                hits_truth.append(is_major)

            # write X and Y files
            Xwriter.writerow(hits_pos)
            Ywriter.writerow(hits_truth)


    X_open.close()
    Y_open.close()
    return extractor_train_dir, X_file, Y_file

def make_data_from_distribution(track_dir, mean, std, windowNum):

    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 9

    ### Construct Path Objects
    dp_name = "dig.mu2e.CeEndpoint.MDC2018b.001002_00000192.art"
    db_file = track_dir.joinpath(dp_name+".db")
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    extractor_train_dir = data_dir.joinpath('extractor_train')
    extractor_train_dir.mkdir(parents=True, exist_ok=True)

    X_name = "hit_pos_train.csv"
    Y_name = "hit_truth_train.csv"
    X_file = extractor_train_dir.joinpath(X_name)
    Y_file = extractor_train_dir.joinpath(Y_name)

    X_open = open(X_file, 'w+')
    Xwriter = csv.writer(X_open, delimiter=',')
    Y_open = open(Y_file, 'w+')
    Ywriter = csv.writer(Y_open, delimiter=',')

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
    bbox_table_row_num = 0
    dict_for_df = {}
    for idx, track_number in enumerate(track_numbers):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
        if idx+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        # get corresponding number of tracks in this window
        track_box = []
        track_found_number = 0
        while track_found_number < track_number:
            ptcl = next(ptcl_iter)
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
            selected_mcs_z = [ z for x, y, z in selected_mcs_pos ]

            # sort positions by its z coordinate
            pos_z_raw  = dict(zip(selected_mcs_pos, selected_mcs_z))
            pos_z = sorted( pos_z_raw.items(), key=lambda item: item[1])

            # justify if hit belongs to the major track
            selected_mcs_pos_flatten = []
            selected_mcs_truth = []
            for pos, z in pos_z:
                is_major = False
                if pos in mcs_pos[i]:
                    is_major = True
                for coordinate in pos:
                    selected_mcs_pos_flatten.append(coordinate)
                selected_mcs_truth.append(is_major)

            # write X and Y files
            Xwriter.writerow(selected_mcs_pos_flatten)
            Ywriter.writerow(selected_mcs_truth)


    X_open.close()
    Y_open.close()
    return extractor_train_dir, X_file, Y_file

def make_data(C, mode):
    pstage("Making training data")


    if mode == 'dp':
        track_dir = C.track_dir
        dp_names = C.source
        window = C.window

        dpNum = len(dp_names)

        for idx, dp_name in enumerate(dp_names):
            pinfo(f"making raw data for data product {idx+1}/{dpNum}: {dp_name}")
            if idx == 0:
                mode = 'first'
            else:
                mode = 'append'
            train_dir, X_file, Y_file = make_data_from_dp(track_dir, dp_name, window, mode)
    elif mode == "normal":
        track_dir = C.track_dir
        mean = C.trackNum_mean
        std = C.trackNum_std
        windowNum = C.window
        train_dir, X_file, Y_file = make_data_from_distribution(track_dir, mean, std, windowNum)

    C.set_inputs(train_dir, X_file, Y_file)
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('extractor.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
    return C

if __name__ == "__main__":
    pbanner()
    psystem('CNN track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    track_str = '/home/Billy/Mu2e/analysis/MLTracking/tracks'
    track_dir = Path(track_str)
    C = Config(track_dir)

    mode = 'normal'
    window = 400 # unit: number of windows
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
