import sys
from pathlib import Path

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *

class Config:
    def __init__(self, track_sql_dir):
        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        # source member
        self.track_dir = track_sql_dir
        self.source = None
        self.window = None

        # alternative source: distribution
        self.trackNum_mean = None
        self.trackNum_std = None

        # input member
        self.train_dir = None
        self.X_file = None
        self.Y_file = None

        # max length for input_arrays
        self.sequence_max_length = None

        # input numpy array for training
        self.X_npy = None
        self.Y_npy = None

        # model and record_name
        self.model_name = None
        self.record_name = None

        # test data
        self.test_source = None
        self.test_dir = None
        self.test_X_file = None
        self.test_Y_file_reference = None
        self.test_Y_file_prediction = None

    def set_source(self, source):
        for file in source:
            assert Path.exists(self.track_dir.joinpath(file+'.db')), \
                t_error(f'{file}.db does not exist')
        self.source = source

    def set_distribution(self, mean, std):
        self.trackNum_mean = mean
        self.trackNum_std = std

    def set_window(self, window):
        self.window = window

    def set_inputs(self, extractor_train_dir, X_file, Y_file):
        self.train_dir = extractor_train_dir
        self.X_file = X_file
        self.Y_file = Y_file

    def set_max_length(self, length):
        self.sequence_max_length = length

    def set_input_array(self, X_npy, Y_npy):
        self.X_npy = X_npy
        self.Y_npy = Y_npy

    def set_outputs(self, model_name, record_name):
        self.model_name = model_name
        self.record_name = record_name

    def set_test_source(self, source):
        for file in source:
            assert Path.exists(self.track_dir.joinpath(file+'.db')), \
                t_error(f'{file}.db does not exist')
        self.test_source = source

    def set_test_data(self, extractor_test_dir, X_file, Y_file):
        self.test_dir = extractor_test_dir
        self.test_X_file = X_file
        self.test_Y_file_reference = Y_file

    def set_prediction(self, prediction_file):
        self.test_Y_file_prediction = prediction_file
