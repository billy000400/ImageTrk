# Detector (Faster RCNN)
# forward propogate from input to output
# Goal: test if the validation output act as expected

import sys
from pathlib import Path
import pickle
import timeit
from datetime import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, TimeDistributed, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.metrics import CategoricalAccuracy

script_dir = Path.cwd().parent.parent.parent.joinpath('frcnn_mc_train')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config
from DataGenerator import DataGeneratorV2
from Layers import *
from Loss import *
from Metric import *


# load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

# re-build model
input_layer = Input(C.input_shape)
base_net = C.base_net.get_base_net(input_layer, trainable=False)
rpn_layer = rpn(C.anchor_scales, C.anchor_ratios)
classifier = rpn_layer.classifier(base_net)
regressor = rpn_layer.regression(base_net)
model = Model(inputs=input_layer, outputs = [classifier,regressor])
model.summary()

# load model weights
model.load_weights(str(Path.cwd().joinpath('rpn_mc_00.h5')), by_name=True)

# set data generator
val_generator = DataGeneratorV2(C.validation_img_inputs_npy, C.validation_labels_npy, C.validation_deltas_npy, batch_size=8)

# evaluate model
rpn_class_loss = define_rpn_class_loss(1)
rpn_regr_loss = define_rpn_regr_loss(100)
adam = Adam()

class StdCallback(tf.keras.callbacks.Callback):
    accs = []
    ious = []

    def on_test_batch_end(self, batch, logs=None):
        self.accs.append(logs['rpn_out_class_unmasked_binary_accuracy'])
        self.ious.append(logs['rpn_out_regress_unmasked_IoU'])

    def on_test_end(self, epoch, logs=None):
        accs = np.array(self.accs)
        ious = np.array(self.ious)
        print()
        print(f'accs_std:{accs.std()}; ious_std:{ious.std()}')

model.compile(optimizer=adam, loss={'rpn_out_class' : rpn_class_loss,\
                                        'rpn_out_regress': rpn_regr_loss},\
                                    metrics={'rpn_out_class': [unmasked_binary_accuracy, positive_number],\
                                             'rpn_out_regress': unmasked_IoU})

result = model.evaluate(x=val_generator, return_dict=True, callbacks=[StdCallback()])
result = {key:[value] for key, value in result.items()}
df = pd.DataFrame.from_dict(result)
df.to_csv(Path.cwd().joinpath('result.csv'), index=None)
