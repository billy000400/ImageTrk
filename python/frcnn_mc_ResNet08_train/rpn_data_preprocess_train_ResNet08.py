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
from Architectures import ResNet08
from Information import*


def preprocess(C):

    pstage('Preprocess Data')

    # Check if raw taining set is created
    assert C.input_shape != None, \
        t_error("You have to make training set before preprocessing it")

    assert (C.label_limit_lower != None) and (C.label_limit_upper != None),\
        t_error('You have to setup rpn label limits before precrocessing data')

    # unpacking parameters
    lim_lo = C.label_limit_lower
    lim_up = C.label_limit_upper

    # setup save path for preprocessed data
    cwd = Path.cwd()
    tmp_dir = C.data_dir.parent.joinpath('tmp')
    C.tmp_dir = tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np_dir = tmp_dir

    # Get anchors
    anchors = make_anchors(C.input_shape, C.base_net.ratio, C.anchor_scales, C.anchor_ratios) # anchors have been normalized

    # Get bbox dicts. A bbox dict is {img_name: bboxes_list}
    pinfo('Making the image-bbox dictionary')
    img_bbox_dict = make_img_bbox_dict(C.img_dir, C.bbox_reference_file)

    # loop through img_bbox list
    inputs = []
    label_maps = []
    delta_maps = []
    img_bbox_list = [ [img_name, bbox_list] for img_name, bbox_list in img_bbox_dict.items() ]
    bbox_Num = len(pd.read_csv(C.bbox_reference_file, index_col=0).index)
    bbox_idx = 0
    for img_name, bbox_list in img_bbox_list:
        # get input
        img_path_str = str(C.img_dir.joinpath(img_name))
        input = cv2.imread(img_path_str)
        if input.any() == None:
            perr(f'{img_path_str} is invalid')
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
            inputs.append(input)
            label_maps.append(sampled_label_map)
            delta_maps.append(delta_map)
        else:
            pwarn(f'{img_name} is discarded as it has untrainable data', special = '\n')
            pwarn(f'Detalis: labels_trainable:{labels_trainable}, deltas_trainable:{deltas_trainable}')
            df = pd.read_csv(C.bbox_reference_file, index_col=0)
            df = df[df['FileName']!=img_name]
            df.to_csv(C.bbox_reference_file)

    # Save numpy arrays to local
    inputs = np.asarray(inputs)
    label_maps = np.asarray(label_maps)
    delta_maps = np.asarray(delta_maps)

    inputs_npy = np_dir.joinpath('mc_inputs.npy')
    labels_npy = np_dir.joinpath('mc_label_maps.npy')
    deltas_npy = np_dir.joinpath('mc_delta_maps.npy')

    inputs = inputs/255.0

    np.save(inputs_npy, inputs)
    np.save(labels_npy, label_maps)
    np.save(deltas_npy, delta_maps)

    # setup configuration
    C.set_rpn_training_data(inputs_npy, labels_npy, deltas_npy)

    pickle_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_path, 'wb'))

    pcheck_point('Preprocessed data')
    return C


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    pinfo('Parameters are set inside the script')

    # configure important parameters
    base_nn_name = 'vgg16'
    anchor_scales = [0.1, 0.3, 0.8]
    anchor_ratios = [[1,1],\
                        [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)]]
    lower_limit = 0.3
    upper_limit = 0.7

    posCut = 6
    nWant = 12

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))
    vgg16  = ResNet08()
    C.set_base_net(vgg16)
    C.set_anchor(anchor_scales, anchor_ratios)
    C.set_label_limit(lower_limit, upper_limit)
    C.set_sample_parameters(posCut, nWant)



    start = timeit.default_timer()
    preprocess(C)
    total_time = timeit.default_timer()-start
    print('\n')
    pinfo(f'Elapsed time: {total_time}(sec)')
