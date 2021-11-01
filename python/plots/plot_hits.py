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
from Database_new import *
from Configuration import frcnn_config
from Abstract import binning_objects
from Information import *

def rotate(xs, ys, faces, angle_per_face):
    xrs = []
    yrs = []
    for x,y,f in zip(xs,ys,faces):
        sin = np.sin(f*angle_per_face)
        cos = np.cos(f*angle_per_face)
        v = np.array([x,y])
        v.shape = (2,1)
        R = np.matrix([[cos, -sin],[sin, cos]])
        v_r = R.dot(v)
        xr = v_r[0]
        yr = v_r[1]
        xrs.append(xr)
        yrs.append(yr)

    return xrs, yrs


def plotHits(db_file, angle):
    mean = 5
    std = 3
    windowNum = 5
    resolution = 256

    hitNumCut = 20

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
    for idx, track_number in enumerate(track_numbers):
        sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
        if idx+1 == windowNum:
            sys.stdout.write('\n')
        sys.stdout.flush()

        x_all = []
        y_all = []
        uniqueFace_all = []

        track_found_num = 0

        while track_found_num < track_number:
            ptcl = next(ptcl_iter)

            strawHit_qrl = session.query(StrawDigiMC).filter(StrawDigiMC.particle==ptcl.id)
            hitNum = strawHit_qrl.count()
            if (hitNum >= hitNumCut) and (ptcl.pdgId == 11):
                strawHits = strawHit_qrl.all()
                xs = [hit.x for hit in strawHits]
                ys = [hit.y for hit in strawHits]
                uniqueFaces = [hit.uniqueFace for hit in strawHits]

                x_all.append(xs)
                y_all.append(ys)
                uniqueFace_all.append(uniqueFaces)

                track_found_num += 1
            else:
                continue

        plt.figure(figsize=(8,8), frameon=False)
        print("Number of particles: ",len(x_all))
        for xs, ys, fs in zip(x_all, y_all, uniqueFace_all):
            xrs, yrs = rotate(xs, ys, fs, angle)
            plt.scatter(xrs, yrs, s=1)
            plt.axis('scaled')
            plt.axis('off')
            plt.xlim([-810, 810])
            plt.ylim([-810, 810])
        plt.show()
        plt.close()

    return


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Plotting')

    # initialize parameters
    track_dir_str = '../../tracks'
    track_dir = Path(track_dir_str)
    track_file = track_dir.joinpath('test.db')
    plotHits(track_file, -np.pi/6)
