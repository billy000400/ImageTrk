# @Author: Billy Li <billyli>
# @Date:   01-11-2022
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 01-12-2022



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

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from HitGenerators import Event_V2 as Event
from Configuration import wcnn_config
from DataGenerator import DataGeneratorV2
from Architectures import Img2Vec
from Loss import *
from Metric import *

if __name__ == "__main__":
    cwd = Path.cwd()
    pickle_path = cwd.joinpath('wcnn.config.pickle')
    C = pickle.load(open(pickle_path,'rb'))

    # re-build model
    model = Img2Vec(input_shape=(256, 256, 1)).get_model()
    # model.summary()

    # load model weights
    weight_file_name = 'wcnn_01_weight_1_10.h5'
    print(Path.cwd().joinpath(weight_file_name))
    # model.load_weights(Path.cwd().joinpath(weight_file_name))
    model.summary()
    # set data generator
    val_generator = DataGeneratorV2(C.X_train_dir, C.Y1_train_dir, C.Y2_train_dir, batch_size=1)

    # evaluate model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    adam = Adam(learning_rate=lr_schedule)

    class StdCallback(tf.keras.callbacks.Callback):
        accs = []

        def on_test_batch_end(self, batch, logs=None):
            self.accs.append(logs['top2_categorical_accuracy'])

        def on_test_end(self, epoch, logs=None):
            accs = np.array(self.accs)
            print()
            print(f'accs_std:{accs.std()}')

    classifier_loss = define_rpn_class_loss(1, weight=C.weights)
    regressor_loss = define_rpn_regr_loss(10)

    model.compile(optimizer=adam, loss={'classifier': classifier_loss,\
                                        'regressor': regressor_loss},
                                    metrics={'classifier': [unmasked_precision, unmasked_recall, positive_number],\
                                            'regressor': unmasked_IoU1D})
    model.load_weights(Path.cwd().joinpath(weight_file_name))

    # for layer in model.layers: print(layer.get_config(), layer.get_weights())
    result = model.evaluate(x=val_generator)
    # result = {key:[value] for key, value in result.items()}
    # df = pd.DataFrame.from_dict(result)
    # df.to_csv(Path.cwd().joinpath('result.csv'), index=None)
