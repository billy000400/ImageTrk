from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.initializers import RandomNormal

initializer = RandomNormal(stddev=0.01)

class rpn():
    def __init__(self, anchor_scales, anchor_ratios):
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.shared_layer = Conv2D(512, (3, 3), padding='same', activation = 'relu', kernel_initializer=initializer, name='rpn_conv1')

    def classifier(self, base):
        x = self.shared_layer(base)
        x_class = Conv2D(self.num_anchors, (1, 1), activation='sigmoid', kernel_initializer=initializer, name='rpn_out_class')(x)
        return x_class

    def regression(self, base):
        x = self.shared_layer(base)
        x_regr = Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer=initializer, name='rpn_out_regress')(x)
        return x_regr
