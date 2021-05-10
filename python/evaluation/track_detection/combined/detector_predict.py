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
from tensorflow.keras.metrics import CategoricalAccuracy

script_dir = Path.cwd().parent.parent.parent.joinpath('frcnn_mc_train')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Information import *
from Configuration import frcnn_config
from DataGenerator import DataGeneratorV3
from Layers import *
from Loss import *
from Metric import *


# load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('frcnn.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

# re-build model
oneHotEncoder = C.oneHotEncoder
classNum = len(oneHotEncoder)
img_input = Input(shape=C.input_shape)
RoI_input = Input(shape=(C.roiNum, 4))
x = C.base_net.get_base_net(img_input, trainable=False)
x = RoIPooling(6,6)([x, RoI_input])
x = TimeDistributed(Flatten(name='flatten'))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fcl', trainable=False))(x)
x = TimeDistributed(Dense(4096, activation='relu', name='fc2', trainable=False))(x)
x1 = TimeDistributed(Dense(classNum, name='fc3', trainable=False))(x)
x2 = TimeDistributed(Dense(classNum*4, activation='relu', name='fc4', trainable=False))(x)
output_classifier = TimeDistributed(Softmax(axis=-1), name='detector_out_class')(x1)
output_regressor = TimeDistributed(Dense(classNum*4, activation='linear', trainable=False), name='detector_out_regr')(x2)

model = Model(inputs=[img_input, RoI_input], outputs = [output_classifier, output_regressor])
model.summary()

# load model weights
model.load_weights(str(Path.cwd().joinpath('rpn_mc_00.h5')), by_name=True)
model.load_weights(str(Path.cwd().joinpath('detector_mc_RCNN_dr=0.0.h5')), by_name=True)

# set data generator
val_generator = DataGeneratorV3(C.validation_img_inputs_npy, C.validation_rois,\
                                    C.detector_validation_Y_classifier,\
                                    C.detector_validation_Y_regressor,\
                                    shuffle=False,\
                                    batch_size=1)

# evaluate model
adam = Adam()
detector_class_loss = define_detector_class_loss(1)
detector_regr_loss = define_detector_regr_loss(100)
ca = CategoricalAccuracy()
model.compile(optimizer=adam, loss={'detector_out_class':detector_class_loss,\
                                    'detector_out_regr':detector_regr_loss},\
                                metrics = {'detector_out_class':ca,\
                                            'detector_out_regr':unmasked_IoUV2})


class StdCallback(tf.keras.callbacks.Callback):
    accs = []
    ious = []

    def on_test_batch_end(self, batch, logs=None):
        self.accs.append(logs['detector_out_class_categorical_accuracy'])
        self.ious.append(logs['detector_out_regr_unmasked_IoUV2'])

    def on_test_end(self, epoch, logs=None):
        accs = np.array(self.accs)
        ious = np.array(self.ious)
        print()
        print(f'accs_std:{accs.std()}; ious_std:{ious.std()}')

result = model.evaluate(x=val_generator, return_dict=True, callbacks=[StdCallback()])
result = {key:[value] for key, value in result.items()}
df = pd.DataFrame.from_dict(result)
df.to_csv(Path.cwd().joinpath('result.csv'), index=None)
