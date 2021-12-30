# @Author: Billy Li <billyli>
# @Date:   08-03-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 09-19-2021



## @package Config
# Hello this is me
#
# more details

# Default Configure Setup
import sys
from pathlib import Path

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Information import *

class frcnn_config:

    def __init__(self, track_sql_dir, data_dir):

        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        data_dir.mkdir(parents=True, exist_ok=True)

        ############### rpn training parameters starts ###############
        #### make rpn training data

        ## the directory where the track databases are located
        self.track_dir = track_sql_dir
        self.data_dir = data_dir
        self.sub_data_dir = None
        self.tmp_dir = None
        self.weight_dir = None
        self.train_dp_list = None
        self.val_dp_list = None

        ## raw data generation information
        # mode 1: constant sampling window
        self.source = None
        # mode 2: draw tracks from a Gaussian distribution
        self.trackNum_mean = None
        self.trackNum_std = None
        # parameters shared by both modes
        self.window = None # =sampling time in mode 1; trackNum in mode 2
        self.resolution = None

        ## raw data location information
        self.train_img_dir = None
        self.train_bbox_reference_file = None
        self.validation_img_dir = None
        self.validation_bbox_reference_file = None
        self.input_shape = None

        #### preprocess rpn training data

        ## set basenet class
        self.base_net = None # you need to use get_base_net() to get the base_net

        ## Anchor box scales
        self.anchor_scales = None
        self.anchor_ratios = None

        ## rpn score limits for labeling
        self.label_limit_lower = None
        self.label_limit_upper = None

        ## rpn label number limits for bias prevention in training
        self.pos_lo_limit = None
        self.tot_lo_limit = None

        ## set preprocessed data directory
        self.train_img_inputs_npy = None # also used in the detector training
        self.train_labels_npy = None
        self.train_deltas_npy = None

        self.validation_img_inputs_npy = None # also used in the detector training
        self.validation_labels_npy = None
        self.validation_deltas_npy = None

        #### rpn training parameters

        ## regularizations
        self.rpn_lambdas = None

        ## RPN model and training record_name
        self.rpn_model_name = None
        self.rpn_record_name = None

        ############### rpn training parameters ends ###############
        ############### rpn to roi parameters starts ###############

        ## prediction on trained data for detector training
        self.train_bbox_proposal_file = None
        self.validation_bbox_proposal_file = None

        ############### rpn to roi parameters ends ###############
        ############### detector training parameters starts ###############

        # RoI parameters
        self.roiNum = None
        self.negativeRate = None

        # oneHotEncoder reference
        self.oneHotEncoder = None

        # detector training data
        self.train_rois = None
        self.detector_train_Y_classifier = None
        self.detector_train_Y_regressor = None

        self.validation_rois = None
        self.detector_validation_Y_classifier = None
        self.detector_validation_Y_regressor = None

        # detector training regulation
        self.detector_lambda = None

        # detector trianing record
        self.detector_model_name = None
        self.detector_record_name = None

        # detector predcition
        self.validation_bbox_prediction_file = None

        ############### detector training parameters ends ###############
        ############### alternative training parameters starts ###############
        self.frcnn_model_name = None
        self.frcnn_record_name = None
        ############### alternative training parameters ends ###############
        ############### testing parameters starts ###############
        self.test_img_dir = None
        self.test_inputs_npy = None
        self.test_bbox_reference_file = None
        self.test_bbox_proposal_file = None

        self.test_detector_imgs_npy_file = None
        self.test_detector_rois_file = None

        self.test_bbox_prediction_file = None
        ############### testing data parameters starts ###############


    ### RPN training: make raw data
    def set_train_dp_list(self, dp_list):
        self.train_dp_list = dp_list

    def set_val_dp_list(self, dp_list):
        self.val_dp_list = dp_list

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

    def set_raw_training_data(self, bbox_file, img_dir, depth=3):
        self.input_shape = (self.resolution, self.resolution, depth)
        self.train_bbox_reference_file = bbox_file
        self.train_img_dir = img_dir
        return

    def set_raw_validation_data(self, bbox_file, img_dir):
        self.validation_bbox_reference_file = bbox_file
        self.validation_img_dir = img_dir
        return

    ### RPN training: preprocess data
    def set_base_net(self, base_net_class):
        try:
            #base_net = base_net_class.get_base_net(Input(shape=self.input_shape))
            self.base_net = base_net_class
        except:
            perr(f"Invalid base net {base_net_class}")
            sys.exit()
        return

    def set_anchor(self, scales, ratios):
        self.anchor_scales = scales
        self.anchor_ratios = ratios

    def set_label_limit(self, lim_lo, lim_up):
        self.label_limit_lower = lim_lo
        self.label_limit_upper = lim_up

    def set_sample_parameters(self, pos_lo_limit, tot_lo_limit):
        self.pos_lo_limit = pos_lo_limit
        self.tot_lo_limit = tot_lo_limit

    def set_rpn_training_data(self, inputs_npy, labels_npy, deltas_npy):
        self.train_img_inputs_npy = inputs_npy
        self.train_labels_npy = labels_npy
        self.train_deltas_npy = deltas_npy

    def set_rpn_validation_data(self, inputs_npy, labels_npy, deltas_npy):
        self.validation_img_inputs_npy = inputs_npy
        self.validation_labels_npy = labels_npy
        self.validation_deltas_npy = deltas_npy

    def has_preprocessed(self):
        return (self.train_labels_npy!=None) and (self.validation_labels_npy!=None)

    ### RPN training: training
    def set_rpn_lambda(self, lambdas):
        self.rpn_lambdas = lambdas

    def set_rpn_record(self, model_name, record_name):
        self.rpn_model_name = model_name
        self.rpn_record_name = record_name

    ### RPN to RoI start
    def set_train_proposal(self, bbox_proposal_file):
        self.train_bbox_proposal_file = bbox_proposal_file

    def set_validation_proposal(self, bbox_proposal_file):
        self.validation_bbox_proposal_file = bbox_proposal_file

    ### Detector training: make data
    def set_roi_parameters(self, roiNum, negativeRate):
        self.roiNum = roiNum
        self.negativeRate = negativeRate

    def set_oneHotEncoder(self, oneHotEncoder):
        self.oneHotEncoder = oneHotEncoder

    def set_detector_training_data(self, rois, classifier, regressor):
        self.train_rois = rois
        self.detector_train_Y_classifier = classifier
        self.detector_train_Y_regressor = regressor

    def set_detector_validation_data(self, rois, classifier, regressor):
        self.validation_rois = rois
        self.detector_validation_Y_classifier = classifier
        self.detector_validation_Y_regressor = regressor

    ### Detector training: training
    def set_detector_lambda(self, lambdas):
        self.detector_lambda = lambdas
        return

    def set_detector_record(self, model_name, record_name):
        self.detector_model_name = model_name
        self.detector_record_name = record_name

    ### Alternative training
    def set_frcnn_record(self, model_name, record_name):
        self.frcnn_model_name = model_name
        self.frcnn_record_name = record_name
        return

    ### predict validation data by trained weights
    def set_validation_prediction(self, file):
        self.validation_bbox_prediction_file = file

    ### testing: make frcnn/rpn testing data
    def set_testing_data(self, img_dir, inputs_npy, bbox_file):
        self.test_inputs_npy = inputs_npy
        self.test_img_dir = img_dir
        self.test_bbox_reference_file = bbox_file

    ### testing: rpn test
    def set_test_proposal(self, proposal):
        self.test_bbox_proposal_file = proposal

    ### testing: make detector testing data
    def set_detector_testing_data(self, imgs_npy_file, rois_file):
        self.test_detector_imgs_npy_file = imgs_npy_file
        self.test_detector_rois_file = rois_file

    ### testing: detector test
    def set_test_prediction(self, prediction):
        self.test_bbox_prediction_file = prediction

    ### (deprecated) test data
    def set_test_source(self, source):
        for file in source:
            assert Path.exists(self.track_dir.joinpath(file+'.db')), \
                t_error(f'{file}.db does not exist')
        self.test_source = source

    def set_test_distribution(self, mean, std):
        self.test_trackNum_mean = mean
        self.test_trackNum_std = std


class extractor_config:
    def __init__(self, track_sql_dir, data_dir):
        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        # source member
        self.track_dir = track_sql_dir
        self.data_dir = data_dir
        self.sub_data_dir = None

        self.source = None
        self.window = None
        self.resolution = None

        # alternative source: distribution
        self.trackNum_mean = None
        self.trackNum_std = None

        self.train_dp_list = None
        self.val_dp_list = None

        # input member
        self.train_dir = None
        self.iou_dir = None
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

    def set_train_dp_list(self, dp_list):
        self.train_dp_list = dp_list

    def set_val_dp_list(self, dp_list):
        self.val_dp_list = dp_list

    def set_inputs(self, extractor_train_dir, X_file, Y_file):
        self.train_dir = extractor_train_dir
        self.X_file = X_file
        self.Y_file = Y_file

    def set_train_dir(self, train_x_dir, train_y_dir, iou_dir=None):
        self.X_train_dir = train_x_dir
        self.Y_train_dir = train_y_dir
        self.iou_dir=iou_dir

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

class wcnn_config:
    def __init__(self, track_sql_dir, data_dir):
        assert Path.exists(track_sql_dir), \
            t_error('The directory for track SQLits database does not exist')

        # source member
        self.track_dir = track_sql_dir
        self.data_dir = data_dir
        self.sub_data_dir = None

        self.eventNum = None
        self.resolution = None

        # source
        self.train_dp_list = None
        self.val_dp_list = None
        self.train_db_files = None

        # input members
        self.X_train_dir = None
        self.Y1_train_dir = None
        self.Y2_train_dir = None
        self.X_val_dir = None
        self.Y1_val_dir = None
        self.Y2_val_dir = None


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

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_eventNum(self, eventNum):
        self.eventNum = eventNum

    def set_train_dp_list(self, dp_list):
        self.train_dp_list = dp_list
        self.train_db_files = [self.track_dir.joinpath(f'{dp}'+'.db') for dp in dp_list]

    def set_val_dp_list(self, dp_list):
        self.val_dp_list = dp_list
        self.val_db_files = [self.track_dir.joinpath(f'{dp}'+'.db') for dp in dp_list]


    def set_train_dir(self, train_x_dir, train_y1_dir, train_y2_dir):
        self.X_train_dir = train_x_dir
        self.Y1_train_dir = train_y1_dir
        self.Y2_train_dir = train_y2_dir

    def set_val_dir(self, val_x_dir, val_y1_dir):
        self.X_val_dir = val_x_dir
        self.Y1_val_dir = val_y1_dir
        self.Y2_val_dir = val_y2_dir

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
