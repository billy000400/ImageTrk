## @package Config
# Hello this is me
#
# more details

# Default Configure Setup
import sys
from pathlib import Path

from tensorflow.keras.layers import Conv2D

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *

class frcnn_config:

    def __init__(self, track_sql_dir):

        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        # data product source and tracking window info

        ## @var track_dir
        # the directory where the track databases are located
        self.track_dir = track_sql_dir
        self.source = None
        self.window = None
        self.resolution = None

        # alternative source: distribution
        self.trackNum_mean = None
        self.trackNum_std = None

        # input
        self.img_dir = None
        self.bbox_file = None
        self.input_shape = None

        # set base network
        self.base_net = None

        # Anchor box scales
        self.anchor_scales = None
        self.anchor_ratios = None

        # rpn score limits for labeling
        self.label_limit_lower = None
        self.label_limit_upper = None

        # rpn label number limits for bias prevention in training
        self.pos_lo_limit = None
        self.tot_lo_limit = None

        # set preprocessed data directory
        self.inputs_npy = None
        self.labels_npy = None
        self.deltas_npy = None

        # regularizations
        self.lambdas = None

        # model and record_name
        self.model_name = None
        self.record_name = None

        # prediction on trained data for trainings of Fast RCNN
        self.bbox_prediction_file = None

        # test data
        self.test_source = None

        self.test_trackNum_mean = None
        self.test_trackNum_std = None

        self.test_img_dir = None
        self.test_bbox_table_reference = None
        self.test_bbox_table_prediction = None

        # detector training data
        self.rois = None
        self.detector_train_Y_classifier = None
        self.detector_train_Y_regressor = None

        # detector trianing record
        self.detector_model_name = None
        self.detector_record_name = None

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

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_inputs(self, bbox_file, img_dir):
        import pandas as pd
        import cv2
        df = pd.read_csv(bbox_file,index_col=0)
        img_names = df['FileName'].unique()
        files = [str(img_dir.joinpath(img)) for img in img_names]
        files_itr = iter(files)
        shape = cv2.imread(next(files_itr)).shape
        for i in range(1,len(files)):
            shape_new = cv2.imread(next(files_itr)).shape
            if shape_new != shape:
                print("[ERROR] Training images' shapes are not consistent")
                raise ValueError
        self.input_shape = shape
        self.bbox_file = bbox_file
        self.img_dir = img_dir
        return shape

    def set_base(self, nn_name):
        try:
            nn_name = nn_name.lower()
        except:
            print("[ERROR]: The base net name should be a string")

        try:
            exec(f"from base_net import {nn_name}")
            exec(f"self.base_net = {nn_name}()")
            pinfo(f'Base net is configured as {nn_name}')
            return self.base_net
        except:
            print("[ERROR]: Invalid base net")

    def set_anchor(self, scales, ratios):
        self.anchor_scales = scales
        self.anchor_ratios = ratios

    def set_label_limit(self, lim_lo, lim_up):
        self.label_limit_lower = lim_lo
        self.label_limit_upper = lim_up

    def set_sample_parameters(self, pos_lo_limit, tot_lo_limit):
        self.pos_lo_limit = pos_lo_limit
        self.tot_lo_limit = tot_lo_limit

    def set_input_array(self, inputs_npy, labels_npy, deltas_npy):
        self.inputs_npy = inputs_npy
        self.labels_npy = labels_npy
        self.deltas_npy = deltas_npy

    def set_rpn(self):
        from frcnn_rpn import rpn
        result = rpn(self.anchor_scales, self.anchor_ratios)
        pinfo('RPN is configured')
        return result

    def set_lambda(self, lambdas):
        self.lambdas = lambdas

    def set_outputs(self, model_name, record_name):
        self.model_name = model_name
        self.record_name = record_name

    def set_test_source(self, source):
        for file in source:
            assert Path.exists(self.track_dir.joinpath(file+'.db')), \
                t_error(f'{file}.db does not exist')
        self.test_source = source

    def set_test_distribution(self, mean, std):
        self.test_trackNum_mean = mean
        self.test_trackNum_std = std

    def set_test_data(self, test_img_dir, test_bbox_table_reference):
        self.test_img_dir = test_img_dir
        self.test_bbox_table_reference = test_bbox_table_reference

    def set_prediction(self, bbox_table_prediction_file):
        self.bbox_prediction_file = bbox_table_prediction_file

    def set_detector_training_data(self, rois, classifier, regressor):
        self.rois = rois
        self.detector_train_Y_classifier = classifier
        self.detector_train_Y_regressor = regressor

    def set_detector_record(self, model_name, record_name):
        self.detector_model_name = model_name
        self.detector_record_name = record_name

class extractor_config:
    def __init__(self, track_sql_dir):
        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        # source member
        self.track_dir = track_sql_dir
        self.source = None
        self.window = None
        self.resolution = None

        # alternative source: distribution
        self.trackNum_mean = None
        self.trackNum_std = None

        # input member
        self.train_dir = None
        self.X_file = None
        self.Y_file = None

        # alternate input member for large data
        self.X_train_dir = None
        self.Y_train_dir = None
        self.X_val_dir = None
        self.Y_val_dir = None

        # max length for input_arrays
        self.sequence_max_length = None

        # input numpy array for training
        self.X_npy = None
        self.Y_npy = None

        self.weights = None

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

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_inputs(self, extractor_train_dir, X_file, Y_file):
        self.train_dir = extractor_train_dir
        self.X_file = X_file
        self.Y_file = Y_file

    def set_train_dir(self, train_x_dir, train_y_dir):
        self.X_train_dir = train_x_dir
        self.Y_train_dir = train_y_dir

    def set_val_dir(self, val_x_dir, val_y_dir):
        self.X_val_dir = val_x_dir
        self.Y_val_dir = val_y_dir

    def set_max_length(self, length):
        self.sequence_max_length = length

    def set_input_array(self, X_npy, Y_npy):
        self.X_npy = X_npy
        self.Y_npy = Y_npy

    def set_weights(self, weights):
        self.weights = weights

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
