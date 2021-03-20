# A Faster R-CNN approach to Mu2e Tracking
# Compared to 00, the script should make essencial parameters portable and\
# use more Numpy and Pandas code to sandalize data I/O.
# Another goal is to make the base-NN modular for future improvement.
# Author: Billy Haoyang Li
# Email: li000400@umn.edu

### imports starts
import sys
from pathlib import Path
import pickle
import timeit

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import *
from Config import frcnn_config as Config
from Loss import *
from Metric import *
### imports ends

def rpn_train(C):
    pstage("Start Training")

    # prepare the tensorflow.data.DataSet object
    inputs = np.load(C.inputs_npy)
    label_maps = np.load(C.labels_npy)
    delta_maps = np.load(C.deltas_npy)

    # outputs
    cwd = Path.cwd()
    data_dir = cwd.parent.parent.joinpath('data')
    weights_dir = cwd.parent.parent.joinpath('weights')

    model_weights = weights_dir.joinpath(C.model_name+'.h5')
    record_file = data_dir.joinpath(C.record_name+'.csv')

    pinfo('I/O Path is configured')

    # build the model
    rpn = C.set_rpn()

    input_layer = Input(shape=C.input_shape)
    x = C.base_net.nn(input_layer)
    classifier = rpn.classifier(x)
    regressor = rpn.regression(x)
    model = Model(inputs=input_layer, outputs = [classifier,regressor])
    model.summary()
    # model.load_weights(base_weights, by_name=True)

    # setup loss
    rpn_class_loss = define_rpn_class_loss(C.lambdas[0])
    rpn_regr_loss = define_rpn_regr_loss(C.lambdas[1])

    # setup optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    # setup callbacks
    CsvCallback = tf.keras.callbacks.CSVLogger(str(record_file), separator=",", append=False)


    # setup a callback to monitor number of positive anchors
    # if positive anchor number is less than limit, stop training
    class EarlyStoppingAtMinMetric(tf.keras.callbacks.Callback):
        def __init__(self, monitor, lo_limit):
            super(EarlyStoppingAtMinMetric, self).__init__()
            self.monitor = monitor
            self.lo_limit = lo_limit

        def on_train_batch_end(self, batch, logs=None):
            current = logs.get(self.monitor)
            if current < self.lo_limit:
                self.model.stop_training = True

    posNumCallback = EarlyStoppingAtMinMetric('rpn_out_class_positive_number', 30)

    # compile the model
    model.compile(optimizer=adam, loss={'rpn_out_class' : rpn_class_loss,\
                                        'rpn_out_regress': rpn_regr_loss},\
                                    metrics={'rpn_out_class': [unmasked_binary_accuracy, positive_number],\
                                             'rpn_out_regress': unmasked_IoU})

    # initialize fit parameters
    model.fit(x=inputs, y=[label_maps, delta_maps],\
                validation_split=0.25,\
                shuffle=True,\
                batch_size=6, epochs=300,\
                callbacks = [CsvCallback, posNumCallback])


    # pdebug(model_weights)
    #
    # model.load_weights(model_weights, by_name=True)
    outputs_raw = model.predict(x=inputs, batch_size=1)
    score_maps = outputs_raw[0]
    for idx, score_map in enumerate(score_maps):
        #pdebug(score_map)
        #pdebug(score_map.shape)
        #pdebug(score_map[score_map>0.5])
        pdebug(score_map[score_map>0.5].size)
        # if idx == 1:
        #     sys.exit()

    model.save_weights(model_weights, overwrite=True)

    pickle_train_path = Path.cwd().joinpath('frcnn.train.config.pickle')
    pickle.dump(C, open(pickle_train_path, 'wb'))

    pickle_test_path = Path.cwd().parent.joinpath('frcnn_test').joinpath('frcnn.test.config.pickle')
    pickle.dump(C, open(pickle_test_path, 'wb'))

    pcheck_point('Finished Training')
    return C


if __name__ == "__main__":
    pbanner()
    psystem('Faster R-CNN Object Detection System')
    pmode('Training')

    pinfo('Parameters are set inside the script')

    cwd = Path.cwd()
    pickle_path = cwd.joinpath('frcnn.train.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # initialize parameters
    lambdas = [1, 10]
    model_name = 'deep_rpn_mc_00'
    record_name = 'deep_rpn_mc_record_00'

    C.set_outputs(model_name, record_name)
    C.set_lambda(lambdas)
    C = rpn_train(C)
