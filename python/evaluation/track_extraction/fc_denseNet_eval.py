import sys
from pathlib import Path
import csv
import random
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model, initializers, regularizers
from tensorflow.keras.layers import(
    Input,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPool2D,Dropout,
    Flatten,
    TimeDistributed,
    Embedding,
    Reshape,
    Softmax
)
from tensorflow.keras.optimizers import Adam

script_dir = Path.cwd().parent.parent.joinpath('photographic_mc_FCDenseNet')
sys.path.insert(1, str(script_dir))

util_dir = Path.cwd().parent.parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Configuration import extractor_config
from DataGenerator import DataGenerator as Generator
from Architectures import FC_DenseNet
from Information import *
from Loss import *
from Metric import *

# load configuration object
cwd = Path.cwd()
pickle_path = cwd.joinpath('extractor.test.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

# re-build model
input_shape = (C.resolution, C.resolution, 1)
architecture = FC_DenseNet(input_shape, 3, dr=0.1)
model = architecture.get_model()
model.summary()

# load model weights
weight_file_name = 'photographic_mc_arc_FCDense_Dropout_0.1_Unaugmented.h5'
model.load_weights(str(Path.cwd().joinpath(weight_file_name)), by_name=True)

# set data generator
val_generator = Generator(C.X_val_dir, C.Y_val_dir, batch_size=1)

# evaluate model
weights = [1,1,1]
cce = categorical_focal_loss(alpha=weights, gamma=2)
ca = top2_categorical_accuracy
adam = Adam()

class StdCallback(tf.keras.callbacks.Callback):
    accs = []

    def on_test_batch_end(self, batch, logs=None):
        self.accs.append(logs['top2_categorical_accuracy'])

    def on_test_end(self, epoch, logs=None):
        accs = np.array(self.accs)
        print()
        print(f'accs_std:{accs.std()}')

model.compile(optimizer=adam, loss=cce, metrics=ca)
result = model.evaluate(x=val_generator, return_dict=True, callbacks=[StdCallback()])
result = {key:[value] for key, value in result.items()}
df = pd.DataFrame.from_dict(result)
df.to_csv(Path.cwd().joinpath('result.csv'), index=None)
