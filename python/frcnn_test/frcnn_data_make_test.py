# Make Testing Set
# Author: Billy Haoyang Li

# General import
import sys
from pathlib import Path
import shutil
import timeit
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from TrackDB_Classes import *
from frcnn_config import Config
from frcnn_util import binning_objects
from mu2e_output import *

def make_data_from_dp(track_dir, dp_name, window, resolution, mode='first'):

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


    img_dir = data_dir.joinpath('imgs_test')
    if mode == 'first':
        shutil.rmtree(img_dir, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        img_name_base = 0
    else:
        img_name_base = len([i for i in img_dir.iterdir()])

    csv_name = "bbox_proposal_test_reference.csv"
    bbox_file = data_dir.joinpath(csv_name)

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

    ### Making images
    # group hits in windows by their time
    pinfo('Making images')
    hits = session.query(StrawHit).all()
    hit_times = [hit.t_reco for hit in hits]
    hit_groups = binning_objects(hits, hit_times, wds)
    groupNum = len(hit_groups)

    # make image for each group
    img_list = []
    for idx, group in enumerate(hit_groups):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{groupNum}', special='\r'))
        if idx+1 == groupNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        xs = [hit.x_reco for hit in group]
        ys = [hit.y_reco for hit in group]
        plt.figure(figsize=(resolution/200,resolution/200), dpi=200, frameon=False)
        plt.scatter(xs, ys, c='b', s=0.2, alpha=0.3)
        plt.axis('scaled')
        plt.axis('off')
        plt.xlim([-810, 810])
        plt.ylim([-810, 810])
        img_name = str(idx+img_name_base+1).zfill(5)+'.png'
        img_list.append(img_name)
        img_file = img_dir.joinpath(img_name)
        plt.savefig(img_file)
        plt.close()

    # make hit group dictionary for reference
    hitId_groupIdx_dict = {}
    for idx, group in enumerate(hit_groups):
        for hit in group:
            hitId_groupIdx_dict[hit.id] = idx

    ### Making bbox table
    pinfo('Making the bounding box table')
    dict_for_df = {}
    index = 0
    ptcl_qry = session.query(Particle.id)
    ptcl_ids = [ptcl for ptcl, in ptcl_qry]
    num_ptcl = ptcl_qry.count()
    for idx, ptcl_id in enumerate(ptcl_ids):
        sys.stdout.write(t_info(f'Parsing particles {idx+1}/{num_ptcl}', special='\r'))
        if idx+1 == num_ptcl:
            sys.stdout.write('\n')
        sys.stdout.flush()
        strawHits_qrl = session.query(StrawHit).filter(StrawHit.particle==ptcl_id)
        hitNum = strawHits_qrl.count()
        # check num of hit belongs to the ptcl
        if hitNum < hitNumCut:
            continue
        # understand which pictures(groups) hits belong
        strawHits = strawHits_qrl.all()
        duplicate_img_list = [hitId_groupIdx_dict[strawHit.id] for strawHit in strawHits]
        # understand how frequent an image is(how many hits belong to it)
        cnts = Counter(duplicate_img_list).most_common()
        # select images which have more than hitNumCut hits
        imgs = [img for img, frequency in cnts if frequency >= hitNumCut] # notice that img == group index
        for img in imgs:
            img_name = str(img+img_name_base+1).zfill(5)+'.png'
            strawHits_in_img = [ hit for hit in strawHits if (hitId_groupIdx_dict[hit.id]==img) ]
            x_all = [hit.x_reco for hit in strawHits_in_img]
            y_all = [hit.y_reco for hit in strawHits_in_img]
            x_all = np.array(x_all)
            y_all = np.array(y_all)
            XMin = x_all.min()
            XMax = x_all.max()
            YMin = y_all.min()
            YMax = y_all.max()
            xmin = (XMin+810)/1620
            xmax = (XMax+810)/1620
            ymin = (YMin+810)/1620
            ymax = (YMax+810)/1620
            pdgId = session.query(Particle.pdgId).filter(Particle.id==ptcl_id).one_or_none()[0]
            dict_for_df[index] = {'FileName':img_name,\
                                    'XMin':xmin,\
                                    'XMax':xmax,\
                                    'YMin':ymin,\
                                    'YMax':ymax,\
                                    'ClassName':pdgId}
            index += 1

    # make the pandas dataframe
    train_df = pd.DataFrame.from_dict(dict_for_df, "index")
    # save the table to local
    if mode == 'first':
        train_df.to_csv(bbox_file)
    else:
        train_df.to_csv(bbox_file, mode='a', header=False)

    return bbox_file, img_dir

def make_data_from_distribution(track_dir, mean, std, windowNum, resolution):

    hitNumCut = 9

    ### Construct Path Objects
    dp_name = "dig.mu2e.CeEndpoint.MDC2018b.001002_00000192.art"
    db_file = track_dir.joinpath(dp_name+".db")
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    data_dir.mkdir(parents=True, exist_ok=True)

    img_dir = data_dir.joinpath('imgs_test')
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "bbox_proposal_test_reference.csv"
    bbox_file = data_dir.joinpath(csv_name)

    ### initialize sqlite session
    # Connect to the database
    pinfo('Connecting to the track database')
    engine = create_engine('sqlite:///'+str(db_file))
    # make session
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine) # session factory
    session = Session() # session object

    # get a distribution of integers
    floats = np.random.normal(loc=mean, scale=std, size=int(windowNum/4))
    float_type_ints = np.around(floats)
    track_numbers = float_type_ints.astype(int)

    # get all particles
    ptcls = session.query(Particle).all()
    ptcl_iter = iter(ptcls)
    for i in range(windowNum*mean):
        next(ptcl_iter)

    # get major tracks for each
    bbox_table_row_num = 0
    dict_for_df = {}
    for idx, track_number in enumerate(track_numbers):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{int(windowNum/4)}', special='\r'))
        if idx+1 == int(windowNum/4):
            sys.stdout.write('\n')
        sys.stdout.flush()

        img_name = str(idx+1).zfill(5)+'.png'
        img_file = img_dir.joinpath(img_name)

        x_all = []
        y_all = []

        track_found_num = 0
        while track_found_num < track_number:
            ptcl = next(ptcl_iter)
            strawHit_qrl = session.query(StrawHit).filter(StrawHit.particle==ptcl.id)
            hitNum = strawHit_qrl.count()
            if hitNum >= hitNumCut:
                strawHits = strawHit_qrl.all()
                xs = [hit.x_reco for hit in strawHits]
                ys = [hit.y_reco for hit in strawHits]
                x_all = x_all + xs
                y_all = y_all + ys

                xs = np.array(xs)
                ys = np.array(ys)

                XMin = xs.min()
                XMax = xs.max()
                YMin = ys.min()
                YMax = ys.max()
                xmin = XMin/1620 + 0.5
                xmax = XMax/1620 + 0.5
                ymin = YMin/1620 + 0.5
                ymax = YMax/1620 + 0.5
                pdgId = session.query(Particle.pdgId).filter(Particle.id==ptcl.id).one_or_none()[0]
                dict_for_df[bbox_table_row_num] = {'FileName':img_name,\
                                        'XMin':xmin,\
                                        'XMax':xmax,\
                                        'YMin':ymin,\
                                        'YMax':ymax,\
                                        'ClassName':pdgId}

                bbox_table_row_num += 1
                track_found_num += 1
            else:
                continue

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        plt.figure(figsize=(8,8), dpi=resolution/8, frameon=False)
        plt.scatter(x_all, y_all, c='b', s=1)
        plt.axis('scaled')
        plt.axis('off')
        plt.xlim([-810, 810])
        plt.ylim([-810, 810])
        plt.savefig(img_file)
        plt.close()

    train_df = pd.DataFrame.from_dict(dict_for_df, "index")
    train_df.to_csv(bbox_file)

    return bbox_file, img_dir

def make_data(C, mode='dp'):

    pstage('Make Test Data')

    if mode == 'dp':
        ### unpack parameters from configuration
        track_dir = C.track_dir
        window = C.window
        resolution = C.resolution
        dp_names = C.source
        dpNum = len(dp_names)

        for idx, dp_name in enumerate(dp_names):
            pinfo(f"making raw data for data product {idx+1}/{dpNum}: {dp_name}")
            if idx == 0:
                mode = 'first'
            else:
                mode = 'append'
            bbox_file, img_dir = make_data_from_dp(track_dir, dp_name, window, resolution, mode)

    elif mode == "normal":
        track_dir = C.track_dir
        mean = C.trackNum_mean
        std = C.trackNum_std
        windowNum = C.window
        resolution = C.resolution
        bbox_file, img_dir = make_data_from_distribution(track_dir, mean, std, windowNum, resolution)
    else:
        perr(f"\"{mode}\" mode is not supported")
        sys.exit()

    ### Setup configurations
    C.set_test_data(img_dir, bbox_file)
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.test.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pcheck_point('Test images and reference bbox table')
    return C

if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Testing')

    # load pickle
    try:
        cwd = Path.cwd()
        pickle_path = cwd.joinpath('frcnn.test.config.pickle')
        C = pickle.load(open(pickle_path,'rb'))
    except:
        perr("Test pickle file doesn't exist! Please train the model before testing it")
        sys.exit()

    # initialize parameters
    mode = 'normal'

    # prepare testing data with time measurement
    start = timeit.default_timer()
    make_data(C, mode)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')
