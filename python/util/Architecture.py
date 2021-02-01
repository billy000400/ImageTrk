from tensorflow import keras
from tensorflow.keras import layers, initializers, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *

# U_Net_like
class U_Net_Like_1024:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256, 512, 1024]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [1024, 512, 256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_256:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_256_Dropout:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/400)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/400)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/400)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/400)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_256_L2:
    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=l2(pow(10,-(3+256.0/filters))))(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=l2(pow(10,-(3+256.0/filters))))(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=l2(pow(10,-(3+256.0/filters))))(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=l2(pow(10,-(3+256.0/filters))))(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_128:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [128, 64, 32]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_128_Dropout:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/256)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/256)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/256)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/256)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_64:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [64, 32]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_32:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(16, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [32]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [32, 16]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_32_Dropout:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(16, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(0.3)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(0.3)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [32, 16]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/16*0.1+0.2)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/16*0.1+0.1)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_32_l2:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(1e-6)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(16, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [32]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init, kernel_regularizer=reg)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [32, 16]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init, kernel_regularizer=reg)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class U_Net_Like_16:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(8, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [16]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [16, 8]:
            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            #x = layers.SpatialDropout2D(filters/450)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model



# U_Net
class U_Net:
    def __init__(self, input_shape, num_class, shrink_times=3):
        if len(input_shape) == 2:
            try:
                self.input_shape = input_shape + (1,)
            except:
                self.input_shape = input_shape + [1]
        elif len(input_shape) == 3:
            self.input_shape = input_shape
        else:
            perr("Invalid Input Shape")

        self.num_class = num_class
        self.shrink_times = shrink_times


    def get_model(self):

        init = initializers.RandomNormal(stddev=0.01)

        input = Input(self.input_shape)

        conv1 = Conv2D(32, 3, padding='same', kernel_initializer=init)(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)
        conv1 = Conv2D(32, 3, padding='same', kernel_initializer=init)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)

        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(64, 3, padding='same', kernel_initializer=init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)
        conv2 = Conv2D(64, 3, padding='same', kernel_initializer=init)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)

        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = Conv2D(128, 3, padding='same', kernel_initializer=init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)
        conv3 = Conv2D(128, 3, padding='same', kernel_initializer=init)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)

        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = Conv2D(256, 3, padding='same', kernel_initializer=init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)
        conv4 = Conv2D(256, 3, padding='same', kernel_initializer=init)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)

        upconv1 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv4)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation('relu')(upconv1)
        merge1 = concatenate([upconv1, conv3], axis=3)

        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(merge1)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)
        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)

        upconv2 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv5)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation('relu')(upconv2)
        merge2 = concatenate([upconv2, conv2], axis=3)

        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(merge2)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)
        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)

        upconv3 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv6)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation('relu')(upconv3)
        merge3 = concatenate([upconv3, conv1],axis=3)

        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(merge3)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)
        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)

        output = Conv2D(self.num_class, 1, activation='softmax', padding='same', kernel_initializer=init)(conv7)

        model = Model(input, output)
        return model

class U_Net_Upsampling:

    def __init__(self, input_shape, num_class, shrink_times=3):
        if len(input_shape) == 2:
            try:
                self.input_shape = input_shape + (1,)
            except:
                self.input_shape = input_shape + [1]
        elif len(input_shape) == 3:
            self.input_shape = input_shape
        else:
            perr("Invalid Input Shape")

        self.num_class = num_class
        self.shrink_times = shrink_times


    def get_model(self):

        init = initializers.RandomNormal(stddev=0.01)

        input = Input(self.input_shape)

        conv1 = Conv2D(32, 3, padding='same', kernel_initializer=init)(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)
        conv1 = Conv2D(32, 3, padding='same', kernel_initializer=init)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)

        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(64, 3, padding='same', kernel_initializer=init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)
        conv2 = Conv2D(64, 3, padding='same', kernel_initializer=init)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)

        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = Conv2D(128, 3, padding='same', kernel_initializer=init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)
        conv3 = Conv2D(128, 3, padding='same', kernel_initializer=init)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)

        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = Conv2D(256, 3, padding='same', kernel_initializer=init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)
        conv4 = Conv2D(256, 3, padding='same', kernel_initializer=init)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)

        upconv1 = UpSampling2D(2)(conv4)
        upconv1 = Conv2D(128, kernel_size=1, strides=1, padding='same', kernel_initializer=init)(upconv1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation('relu')(upconv1)
        merge1 = concatenate([upconv1, conv3], axis=3)

        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(merge1)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)
        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)

        upconv2 = UpSampling2D(2)(conv5)
        upconv2 = Conv2D(64, kernel_size=1, strides=1, padding='same', kernel_initializer=init)(upconv2)
        upconv2 = Activation('relu')(upconv2)
        merge2 = concatenate([upconv2, conv2], axis=3)

        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(merge2)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)
        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)

        upconv3 = UpSampling2D(2)(conv6)
        upconv3 = Conv2D(32, kernel_size=1, strides=1, padding='same', kernel_initializer=init)(upconv3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation('relu')(upconv3)
        merge3 = concatenate([upconv3, conv1],axis=3)

        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(merge3)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)
        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)

        output = Conv2D(self.num_class, 1, activation='softmax', padding='same', kernel_initializer=init)(conv7)

        model = Model(input, output)
        return model

# U_ResNet
class U_ResNet:
    def __init__(self, input_shape, num_class, shrink_times=3):
        if len(input_shape) == 2:
            try:
                self.input_shape = input_shape + (1,)
            except:
                self.input_shape = input_shape + [1]
        elif len(input_shape) == 3:
            self.input_shape = input_shape
        else:
            perr("Invalid Input Shape")

        self.num_class = num_class
        self.shrink_times = shrink_times

    def get_model(self):

        init = initializers.RandomNormal(stddev=0.01)
        input = Input(self.input_shape)

        conv1 = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=init)(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)
        conv1 = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=init)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = SpatialDropout2D(32/512)(conv1)

        residual1 = Conv2D(64, 1, 2, padding='same', kernel_initializer=init)(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)
        conv2 = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=init)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = SpatialDropout2D(64/512)(conv2)

        short_merge1 = Add()([residual1,conv2])
        residual2 = Conv2D(128, 1, 2, padding='same', kernel_initializer=init)(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2))(short_merge1)

        conv3 = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)
        conv3 = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=init)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)

        short_merge2 = Add()([residual2, conv3])
        residual3 = Conv2D(256, 1, 2, padding='same', kernel_initializer=init)(conv3)
        pool3 = MaxPooling2D(pool_size=(2,2))(short_merge2)

        conv4 = Conv2D(256, 3, padding="same", kernel_initializer=init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)
        conv4 = Conv2D(256, 3, padding="same", kernel_initializer=init)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)

        upconv1 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv4)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation('relu')(upconv1)
        merge1 = concatenate([upconv1, conv3], axis=3)

        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(merge1)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)
        conv5 = Conv2D(128, 3, padding='same', kernel_initializer=init)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = SpatialDropout2D(128/512)(conv5)

        upconv2 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv5)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation('relu')(upconv2)
        merge2 = concatenate([upconv2, conv2], axis=3)

        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(merge2)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)
        conv6 = Conv2D(64, 3, padding='same', kernel_initializer=init)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = SpatialDropout2D(64/512)(conv6)

        upconv3 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(conv6)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation('relu')(upconv3)
        merge3 = concatenate([upconv3, conv1],axis=3)

        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(merge3)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)
        conv7 = Conv2D(32, 3, padding='same', kernel_initializer=init)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = SpatialDropout2D(32/512)(conv7)

        output = Conv2D(self.num_class, 1, activation='softmax', padding='same', kernel_initializer=init)(conv7)

        model = Model(input, output)
        return model


# Dense Nets
class Dense:

    def __init__(self, in_shape, num_classes):
        self.input_shape = in_shape
        self.num_classes = num_classes

    def get_model(self):
        inputs = keras.Input(shape=self.input_shape)
        reg = l2(0.01)
        init = RandomNormal(stddev=0.01)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same", kernel_initializer=init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=init)(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=init)(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same", kernel_initializer=init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model
