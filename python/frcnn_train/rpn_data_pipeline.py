from pathlib import Path
import sys
from math import sqrt

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from frcnn_config import Config
from frcnn_data_make_train import make_data
from frcnn_data_preprocess_vgg16 import preprocess
from rpn_train import rpn_train
from mu2e_output import *

if __name__ == "__main__":

    # print infos
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    # default dataset
    ds = {  #### raw data parameters
            'track_dir': '/home/Billy/Mu2e/analysis/data/tracks',\
            'dp_list': [],\
            'window': 20,\
            'resolution': 600,\
            #### preprocess parameters
            'base_nn_name':'vgg16',\
            'anchor_scales': [50, 80, 130],\
            'anchor_ratios': [[1,1],\
                                [sqrt(2), 1/sqrt(2)], [1/sqrt(2), sqrt(2)],\
                                [sqrt(3), 1/sqrt(3)], [1/sqrt(3), sqrt(3)],\
                                [2, 1/2], [1/2, 2]],\
            'label_lower_limit': 0.3,\
            'label_higher_limit': 0.7,\
            'posCut': 128,\
            'nWant': 256}
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000014.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000020.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000149.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000192.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000150.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000024.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000136.art')
    ds['dp_list'].append('dig.mu2e.CeEndpoint.MDC2018b.001002_00000011.art')


    # dataset handling
    if len(sys.argv) == 2:
        dataset_str = sys.argv[1]
    elif len(sys.argv) == 1:
        pass

    # pipeline starts
    track_dir = Path(ds['track_dir'])
    C = Config(track_dir)
    C.set_source(ds['dp_list'])
    C.set_window(ds['window'])
    C.set_resolution(ds['resolution'])
    C = make_data(C)

    C.set_base(ds['base_nn_name'])
    C.set_anchor(ds['anchor_scales'], ds['anchor_ratios'])
    C.set_label_limit(ds['label_lower_limit'], ds['label_higher_limit'])
    C.set_sample_parameters(ds['posCut'], ds['nWant'])
    C = preprocess(C)
