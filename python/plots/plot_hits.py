# Make Training Set
# Author: Billy Haoyang Li

# General import

"""@package docstring
Documentation for this module.

More details.
"""

import sys
from pathlib import Path
import random
import shutil
import timeit
import pickle
from collections import Counter

import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Information import *


def make_data_from_distribution(track_file, ):
    track_dir = C.track_dir
    mean = C.trackNum_mean
    std = C.trackNum_std
    windowNum = C.window
    resolution = C.resolution

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

    img_dir = data_dir.joinpath('mc_imgs_train')
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "mc_bbox_proposal_train.csv"
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
    floats = np.random.normal(loc=mean, scale=std, size=windowNum)
    float_type_ints = np.around(floats)
    track_numbers = float_type_ints.astype(int)

    # get all particles
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

        img_name = str(idx+1).zfill(5)+'.png'
        img_file = img_dir.joinpath(img_name)

        x_all = []
        y_all = []

        track_found_num = 0

        rects = []
        while track_found_num < track_number:
            try:
                ptcl = next(ptcl_iter)
            except:
                sys.stdout.write('\n')
                sys.stdout.flush()
                pinfo('Run out of particles')
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
            if (hitNum >= hitNumCut) and (ptcl.pdgId == 11):
                strawHits = strawHit_qrl.all()
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
            else:
                continue

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


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Plotting')

    # initialize parameters
    track_dir_str = '/home/Billy/Mu2e/analysis/DLTracking/tracks'
    track_dir = Path(track_dir_str)
    track_file = track_dir.joinpath('train.db')
    plotHits(track_file, 60)
