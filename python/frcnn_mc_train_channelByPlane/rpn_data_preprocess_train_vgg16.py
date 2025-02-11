# @Author: Billy Li <billyli>
# @Date:   11-14-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 12-01-2021



import sys
from pathlib import Path
import shutil
import pickle
import cv2
import timeit
from math import sqrt

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import frcnn_config
from Abstract import*
from Architectures import VGG16_bottleNeck
from Information import*


def preprocess(C):

    pstage('Preprocess Data')

    # Check if raw taining set is created
    assert C.input_shape != None, \
        t_error("You have to make training set before preprocessing it")

    assert (C.label_limit_lower != None) and (C.label_limit_upper != None),\
        t_error('You have to setup rpn label limits before precrocessing data')

    if C.has_preprocessed():
        pwarn('You have preprocessed the raw data before! The Untrainable data '
                'has been removed and won\'t be shown this time.')

    # unpacking parameters
    lim_lo = C.label_limit_lower
    lim_up = C.label_limit_upper

    # setup save path for preprocessed data
    cwd = Path.cwd()
    tmp_dir = C.sub_data_dir
    C.tmp_dir = tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)

    np_dir = tmp_dir.joinpath('rpn_train')
    shutil.rmtree(np_dir, ignore_errors=True)
    np_dir.mkdir(parents=True)

    label_dir = np_dir.joinpath('labels_npy')
    shutil.rmtree(label_dir, ignore_errors=True)
    label_dir.mkdir(parents=True)

    delta_dir = np_dir.joinpath('deltas_npy')
    shutil.rmtree(delta_dir, ignore_errors=True)
    delta_dir.mkdir(parents=True)

    # Get anchors
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios) # anchors have been normalized

    # Get bbox dicts. A bbox dict is {arr_name: bboxes_list}
    pinfo('Making the image-bbox dictionary')
    img_bbox_dict = make_img_bbox_dict(C.train_img_dir, C.train_bbox_reference_file)

    # loop through img_bbox list
    img_bbox_list = [ [arr_name, bbox_list] for arr_name, bbox_list in img_bbox_dict.items() ]
    bbox_Num = len(pd.read_csv(C.train_bbox_reference_file, index_col=0).index)
    bbox_idx = 0
    file_idx = 0
    for arr_name, bbox_list in img_bbox_list:

        # make truth table for RPN classifier
        score_bbox_map = make_score_bbox_map(anchors)
        for bbox in bbox_list:
            bbox_idx += 1
            sys.stdout.write(t_info(f'Scoring and labeling anchors by bbox: {bbox_idx}/{bbox_Num}', special='\r'))
            if bbox_idx == bbox_Num:
                sys.stdout.write('\n')
            sys.stdout.flush()
            score_bbox_map = update_score_bbox_map(score_bbox_map, bbox, anchors)

        raw_label_map = make_label_map(score_bbox_map, lim_lo, lim_up)
        sampled_label_map = sample_label_map(raw_label_map, C.pos_lo_limit, C.tot_lo_limit)
        delta_map = make_delta_map(score_bbox_map, lim_up, anchors)
        # Check if both label and delta map have trainable data
        labels_trainable = (~np.isnan(sampled_label_map)).any()
        deltas_trainable = (~np.isnan(delta_map)).any()
        trainable = labels_trainable and deltas_trainable
        if trainable:

            label_file = label_dir.joinpath(f'label_{ str(file_idx).zfill(7) }.npy')
            delta_file = delta_dir.joinpath(f'delta_{ str(file_idx).zfill(7) }.npy')

            if (np.count_nonzero(~np.isnan(delta_map))%4)!=0:
                perr('I found the bug!')
                sys.exit()

            np.save(label_file, sampled_label_map)
            np.save(delta_file, delta_map)

            file_idx += 1

        else:
            pwarn(f'{arr_name} is discarded as it has untrainable data', special = '\n')
            pwarn(f'Details: labels_trainable:{labels_trainable}, deltas_trainable:{deltas_trainable}')
            df = pd.read_csv(C.train_bbox_reference_file, index_col=0)
            df = df[df['FileName']!=arr_name]
            df.to_csv(C.train_bbox_reference_file)

    # setup configuration
    C.set_rpn_training_data(C.train_img_dir, label_dir, delta_dir)

    pcheck_point('Preprocessed data')
    return C


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    pinfo('Parameters are set inside the script')

    # configure important parameters
    anchor_scales = [0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
    anchor_ratios = [[1,1],\
                        [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]
    lower_limit = 0.3
    upper_limit = 0.7

    posCut = 32
    nWant = 64

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    vgg16  = VGG16_bottleNeck()
    C.set_base_net(vgg16)
    C.set_anchor(anchor_scales, anchor_ratios)
    C.set_label_limit(lower_limit, upper_limit)
    C.set_sample_parameters(posCut, nWant)



    start = timeit.default_timer()
    preprocess(C)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')

    pickle_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))
