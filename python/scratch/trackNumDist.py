# @Author: Billy Li <billyli>
# @Date:   11-25-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 11-25-2021

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event

track_dir = Path("../../tracks")
db_files = [track_dir.joinpath('train.db')]

# dist, db_files, hitNumCut=20):
gen = Event(db_files, 10)

windowNum = 1000
trackNums = []
for idx in range(windowNum):
    sys.stdout.write(t_info(f'Parsing windows {idx+1}/{windowNum}', special='\r'))
    if idx+1 == windowNum:
        sys.stdout.write('\n')
    sys.stdout.flush()
    hit_all, track_all = gen.generate(mode='eval')
    trackNum = len(track_all)
    if trackNum > 0:
        trackNums.append(trackNum)

plt.hist(trackNums)
plt.show()
