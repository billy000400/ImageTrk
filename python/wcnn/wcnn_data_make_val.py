import sys
from pathlib import Path
import shutil
import timeit
import pickle
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event
from Configuration import wcnn_config

def discretize(x, min, max, res):
    # return the discretized index of a value given a range and resolution
    step = (max-min)/res
    result = (x-min)//step
    if result == res:
        result = res-1
    return int(result)

def residue(x, min, max, res):
    step = (max-min)/res
    res_val = (x-min)%step
    return res_val

def zt2map(zs, ts, res):
    # return a z-t ptcl number map
    map = np.zeros(shape=(res,res), dtype=float)
    zmin = min(zs)
    zmax = max(zs)
    tmin, tmax = min(ts), max(ts)

    for z, t in zip(zs, ts):
        zIdx = discretize(z, zmin, zmax, res)
        tIdx = discretize(t, tmin, tmax, res)
        map[tIdx, zIdx] += 1

    map_max = map.max()
    map = map/map_max
    return map

def zt2vecs(hitsInTracks, tmin, tmax, res):
    step = (tmax-tmin)/res
    vec1 = np.zeros(shape=(res), dtype=np.float)
    vec2 = np.zeros(shape=(res,2), dtype=np.float)

    for hitPerTrack in hitsInTracks:
        tsPerTrack = [hit[3] for hit in hitPerTrack]
        t_avg = sum(tsPerTrack)/len(tsPerTrack)
        delta_t = max(tsPerTrack)-min(tsPerTrack)
        tIdx = discretize(t_avg, tmin, tmax, res)
        vec1[tIdx] = 1.0

        if vec2[tIdx][1] > delta_t:
            continue
        else:
            vec2[tIdx][0] = residue(t_avg, tmin, tmax, res)/step
            vec2[tIdx][1] = math.log(delta_t/step)

    return [vec1, vec2]

def make_data(C):

    ## unpack some parameters
    track_dir = C.track_dir
    resolution = C.resolution
    data_dir = C.data_dir

    ## construct sub_data dir (where the training data and val data are stored)
    C.sub_data_dir = data_dir.joinpath(Path.cwd().name)
    data_dir = C.sub_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    ### directories for numpy data
    val_x_dir = data_dir.joinpath('X')
    val_y1_dir = data_dir.joinpath('Y1')
    val_y2_dir = data_dir.joinpath('Y2')

    shutil.rmtree(val_x_dir, ignore_errors=True)
    shutil.rmtree(val_y1_dir, ignore_errors=True)
    shutil.rmtree(val_y2_dir, ignore_errors=True)

    val_x_dir.mkdir(parents=True, exist_ok=True)
    val_y1_dir.mkdir(parents=True, exist_ok=True)
    val_y2_dir.mkdir(parents=True, exist_ok=True)

    ## prepare event generator
    # Billy: I'm quite confident that all major tracks(e-) have more than 9 hits
    hitNumCut = 10
    gen = Event(C.val_db_files, hitNumCut=hitNumCut, eventNum=C.eventNum)

    maps = []
    vec1s = []
    vec2s = []
    for i in range(C.eventNum):
        sys.stdout.write(t_info(f'Parsing events: {i+1}/{C.eventNum}', special='\r'))
        if i+1 == C.eventNum:
            sys.stdout.write('\n')
        sys.stdout.flush()
        hit_all, track_all = gen.generate(mode='eval')

        hitsInTracks = []
        table = []
        for trkIdx, hitIdcPdgId in track_all.items():
            hitIdc = hitIdcPdgId[:-1]
            hitsPerTrack = [hit_all[idx] for idx in hitIdc]
            tsPerTrack = [hit[3] for hit in hitsPerTrack]
            delta_t= max(tsPerTrack)-min(tsPerTrack)
            if delta_t > 1000:
                continue
            hitsInTracks.append(hitsPerTrack)

        hits = [hit for hitsPerTrack in hitsInTracks for hit in hitsPerTrack]
        zs = [hit[2] for hit in hits]
        ts = [hit[3] for hit in hits]
        tmin, tmax = min(ts), max(ts)

        map = zt2map(zs, ts, C.resolution)
        [vec1, vec2] = zt2vecs(hitsInTracks, tmin, tmax, C.resolution)

        # fileName = str(i).zfill(5)+'.npy'
        # x_file = train_x_dir.joinpath(fileName)
        # y1_file = train_y1_dir.joinpath(fileName)
        # y2_file = train_y2_dir.joinpath(fileName)
        #
        # np.save(x_file, map)
        # np.save(y1_file, vec1)
        # np.save(y2_file, vec2)

        maps.append(map)
        vec1s.append(vec1)
        vec2s.append(vec2)

    maps = np.concatenate(maps)
    vec1s = np.concatenate(vec1s)
    vec2s = np.concatenate(vec2s)

    maps_file = val_x_dir.joinpath('X.npy')
    vec1s_file = val_y1_dir.joinpath('Y1.npy')
    vec2s_file = val_y2_dir.joinpath('Y2.npy')

    np.save(maps_file, maps)
    np.save(vec1s_file, vec1s)
    np.save(vec2s_file, vec2s)

    C.set_val_dir(val_x_dir, val_y1_dir, val_y2_dir)

    return C

if __name__ == "__main__":
    pbanner()
    psystem('Window-based Convolutional Neural Network')
    pmode('Generating validating data')
    pinfo('Input DType for testing: StrawHit')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('wcnn.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    dp_list = ['val_CeEndpoint-mix']
    C.set_val_dp_list(dp_list)

    start = timeit.default_timer()
    C = make_data(C)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')

    pickle.dump(C, open(pickle_path, 'wb'))
