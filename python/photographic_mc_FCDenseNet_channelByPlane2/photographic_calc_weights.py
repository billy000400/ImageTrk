# This script's purpose is to train a preliminary CNN for tracking by Keras
# Author: Billy Li
# Email: li000400@umn.edu

# import starts
import sys
from pathlib import Path
import pickle

import numpy as np



util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config as Config
from Information import *
### import ends
def calc_weights(Y_dir):
    pinfo('Calculating class weights by median frequency')
    Y_list = [child for child in C.Y_train_dir.iterdir()]
    BgNum = 0
    MajorNum = 0
    fileNum = len(Y_list)
    for i, file in enumerate(Y_list):
        sys.stdout.write(t_info(f'Parsing windows {i+1}/{fileNum}', special='\r'))
        y = np.load(file)
        bgNum = np.count_nonzero( (y==0) )
        majorNum = np.count_nonzero( (y==1) )

        BgNum += bgNum
        MajorNum += majorNum

    pinfo(f'Frequency: major {MajorNum}, bg {BgNum}')
    numArr = np.array([MajorNum, BgNum])
    weights = numArr/np.sqrt(MajorNum*BgNum)
    pinfo(f'Weight array = {weights}')

    return weights


if __name__ == "__main__":
    pbanner()
    psystem('Photographic track extractor')
    pmode('Testing Feasibility')
    pinfo('Input DType for testing: StrawDigiMC')

    # load pickle
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('photographic.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    weights = calc_weights(C)
    C.set_weights(weights)
    pickle.dump(C, open(pickle_path, 'wb'))
