### This script is to evaluate the accuracy and IoU of
### the WCNN on the testing set.
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
