from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import *

class U_Net_Like:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        return
