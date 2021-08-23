from tensorflow import keras
from tensorflow.keras import layers, initializers, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *

### feature extraction networks
class VGG16:
    def __init__(self):
        # same padding, so size only shrinks when pooling
        # pooling by 2 for 4 times: ratio = 2^4 = 16
        self.type = 'VGG16'
        self.ratio = 16
        self.final_chanel = 512

    def get_base_net(self, input_layer, trainable=True):

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(input_layer)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(x)
        x = MaxPooling2D((2,2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
        #x = SpatialDropout2D(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
        #x = SpatialDropout2D(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
        #x = SpatialDropout2D(0.3)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
        #x = SpatialDropout2D(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
        #x = SpatialDropout2D(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
        return x

    def get_model(self, in_shape, num_classes):
        return

class DenseNet:
    def __init__(self, dr=0.0):
        self.type = 'DenseNet'
        self.ratio = 2
        self.final_chanel = 256
        self.dr = dr

    def _add_layer(self,x, k, trainable):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(k, (3,3), padding='same', kernel_initializer=self.init,\
                trainable=trainable)(x)
        if self.dr!=0.0:
            x = SpatialDropout2D(self.dr)(x)
        return x

    def _add_DB(self,x, lyn, k, trainable):
        pls = []
        for i in range(lyn):
            x = self.add_layer(x, k, trainable)
            pls.append(x)
            if len(pls) < 2:
                x = x
            else:
                x = concatenate(pls, axis=3)
        return x

    def add_TD(self,x, m):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(m, (1,1), kernel_initializer=self.init)(x)
        if self.dr!=0.0:
            x = SpatialDropout2D(self.dr)(x)
        x = MaxPooling2D((2,2))(x)
        return x

    def get_base_net(self, input_layer, trainable=True):

        m = 64 # current number of feature maps
        lyns = [4, 8] # layer number; one layer is a set of feature maps
        k = 16 # growth rate of feature map due to concatenation

        # Entry block
        x = layers.Conv2D(m, (3,3), strides=1, padding="same", kernel_initializer=self.init,\
                            trainable=trainable)(input_layer)

        ### [First half of the network: downsampling inputs] ###

        for lyn in lyns:
            skip = x
            x = self.add_DB(x, lyn=lyn, k=k, trainable=trainable)
            x = concatenate([skip, x])
            m += lyn*k

        return x

    def get_model(self, in_shape, num_classes):
        return

class ResNet08:

    def __init__(self):
        # same padding, so size only shrinks when pooling
        # pooling by 2 for 4 times: ratio = 2^4 = 16
        self.type = 'Res08'
        self.ratio = 16
        self.final_chanel = 512

    def get_base_net(self, input_layer, trainable=True):
        init = initializers.RandomNormal(stddev=0.01)

        conv1 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer=init)(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = SpatialDropout2D(64/512)(conv1)
        conv1 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer=init)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = SpatialDropout2D(64/512)(conv1)

        residual1 = Conv2D(128, 1, 2, padding='same', kernel_initializer=init)(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = SpatialDropout2D(128/512)(conv2)
        conv2 = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=init)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = SpatialDropout2D(128/512)(conv2)

        short_merge1 = Add()([residual1,conv2])
        residual2 = Conv2D(256, 1, 2, padding='same', kernel_initializer=init)(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2))(short_merge1)

        conv3 = Conv2D(256, 3, strides=1, padding="same", kernel_initializer=init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)
        conv3 = Conv2D(256, 3, strides=1, padding="same", kernel_initializer=init)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = SpatialDropout2D(128/512)(conv3)

        short_merge2 = Add()([residual2, conv3])
        residual3 = Conv2D(512, 1, 2, padding='same', kernel_initializer=init)(conv3)
        pool3 = MaxPooling2D(pool_size=(2,2))(short_merge2)

        conv4 = Conv2D(512, 3, padding="same", kernel_initializer=init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)
        conv4 = Conv2D(512, 3, padding="same", kernel_initializer=init)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = SpatialDropout2D(256/512)(conv4)

        short_merge_3 = Add()([residual3, conv4])
        residual4 = Conv2D(512, 1, 2, padding='same', kernel_initializer=init)(conv4)
        pool4 = MaxPooling2D(pool_size=(2,2))(short_merge_3)

        conv5 = Conv2D(512, 3, padding="same", kernel_initializer=init)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv5 = SpatialDropout2D(256/512)(conv5)
        conv5 = Conv2D(512, 3, padding="same", kernel_initializer=init)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        #conv5 = SpatialDropout2D(256/512)(conv5)

        short_merge_4 = Add()([residual4, conv5])

        return short_merge_4

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

class U_Net_Like_1024_Dropout:

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
            x = layers.SpatialDropout2D(filters/3414)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/3414)(x)
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
            x = layers.SpatialDropout2D(filters/3414)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/3414)(x)
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

class U_Net_Like_512:

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
        for filters in [64, 128, 256, 512]:
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

        for filters in [512, 256, 128, 64, 32]:
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

class U_Net_Like_512_Dropout:

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
        for filters in [64, 128, 256, 512]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1024)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1024)(x)
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

        for filters in [512, 256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1024)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1024)(x)
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
            x = layers.SpatialDropout2D(filters/1280)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1280)(x)
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
            x = layers.SpatialDropout2D(filters/1280)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/1280)(x)
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
            x = layers.SpatialDropout2D(filters/427)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/427)(x)
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
            x = layers.SpatialDropout2D(filters/427)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SpatialDropout2D(filters/427)(x)
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
class FC_DenseNet:

    def __init__(self, in_shape, num_classes, dr=0.3):
        self.input_shape = in_shape
        self.num_classes = num_classes
        self.init = RandomNormal(stddev=0.01)
        self.dr=dr

    def add_TD(self,x, m):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(m, (1,1), kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        x = MaxPooling2D((2,2))(x)
        return x

    def add_TU(self,x, m):
        x = Conv2DTranspose(m, 3, strides=2, padding='same')(x)
        return x

    def add_layer(self,x, k):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(k, (3,3), padding='same', kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        return x

    def add_DB(self,x, lyn, k):
        pls = []
        for i in range(lyn):
            x = self.add_layer(x, k)
            pls.append(x)
            if len(pls) < 2:
                x = x
            else:
                x = concatenate(pls, axis=3)
        return x

    def get_model(self):

        # parameters
        input_shape = self.input_shape
        m = 48 # current number of feature maps
        lyns = [4, 5, 7, 10, 12] # layer number; one layer is a set of feature maps
        k = 16 # growth rate of feature map due to concatenation

        # input layer
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Conv2D(m, (3,3), strides=1, padding="same", kernel_initializer=self.init)(inputs)

        ### [First half of the network: downsampling inputs] ###
        lskip = []
        for lyn in lyns:
            skip = x
            x = self.add_DB(x, lyn=lyn, k=k)
            x = concatenate([skip, x])
            lskip.append(x)
            m += lyn*k
            x = self.add_TD(x, m=m)

        ### [Bottle neck] ###
        x = self.add_DB(x, lyn=15, k=k)
        m = 15*k

        ### [Second half of the network: upsampling inputs] ###
        lyns_rvs = lyns[::-1]
        lskip_rvs = lskip[::-1]
        for i, lyn in enumerate(lyns_rvs):
            x = self.add_TU(x, m=m)
            x = concatenate([lskip_rvs[i], x])
            x = self.add_DB(x, lyn=lyn, k=k)
            m = lyn*k


        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 1, activation="softmax", padding="same", kernel_initializer=self.init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class FC_DenseNet_largeKernel:

    def __init__(self, in_shape, num_classes, dr=0.3):
        self.input_shape = in_shape
        self.num_classes = num_classes
        self.init = RandomNormal(stddev=0.01)
        self.dr=dr

    def add_TD(self,x, m):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(m, (1,1), kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        x = MaxPooling2D((2,2))(x)
        return x

    def add_TU(self,x, m):
        x = Conv2DTranspose(m, 3, strides=2, padding='same')(x)
        return x

    def add_layer(self,x, k):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(k, (3,3), padding='same', kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        return x

    def add_DB(self,x, lyn, k):
        pls = []
        for i in range(lyn):
            x = self.add_layer(x, k)
            pls.append(x)
            if len(pls) < 2:
                x = x
            else:
                x = concatenate(pls, axis=3)
        return x

    def get_model(self):

        # parameters
        input_shape = self.input_shape
        m = 48 # current number of feature maps
        lyns = [4, 5, 7, 10, 12] # layer number; one layer is a set of feature maps
        k = 16 # growth rate of feature map due to concatenation

        # input layer
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Conv2D(m, (17,17), strides=1, padding="same", kernel_initializer=self.init)(inputs)

        ### [First half of the network: downsampling inputs] ###
        lskip = []
        for lyn in lyns:
            skip = x
            x = self.add_DB(x, lyn=lyn, k=k)
            x = concatenate([skip, x])
            lskip.append(x)
            m += lyn*k
            x = self.add_TD(x, m=m)

        ### [Bottle neck] ###
        x = self.add_DB(x, lyn=15, k=k)
        m = 15*k

        ### [Second half of the network: upsampling inputs] ###
        lyns_rvs = lyns[::-1]
        lskip_rvs = lskip[::-1]
        for i, lyn in enumerate(lyns_rvs):
            x = self.add_TU(x, m=m)
            x = concatenate([lskip_rvs[i], x])
            x = self.add_DB(x, lyn=lyn, k=k)
            m = lyn*k


        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 1, activation="softmax", padding="same", kernel_initializer=self.init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model

class FC_DenseNet_AveragePooling:

    def __init__(self, in_shape, num_classes, dr=0.3):
        self.input_shape = in_shape
        self.num_classes = num_classes
        self.init = RandomNormal(stddev=0.01)
        self.dr=dr

    def add_TD(self,x, m):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(m, (1,1), kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        x = AveragePooling2D((2,2))(x)
        return x

    def add_TU(self,x, m):
        x = Conv2DTranspose(m, 3, strides=2, padding='same')(x)
        return x

    def add_layer(self,x, k):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(k, (3,3), padding='same', kernel_initializer=self.init)(x)
        x = SpatialDropout2D(self.dr)(x)
        return x

    def add_DB(self,x, lyn, k):
        pls = []
        for i in range(lyn):
            x = self.add_layer(x, k)
            pls.append(x)
            if len(pls) < 2:
                x = x
            else:
                x = concatenate(pls, axis=3)
        return x

    def get_model(self):

        # parameters
        input_shape = self.input_shape
        m = 48 # current number of feature maps
        lyns = [4, 5, 7, 10, 12] # layer number; one layer is a set of feature maps
        k = 16 # growth rate of feature map due to concatenation

        # input layer
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Conv2D(m, (3,3), strides=1, padding="same", kernel_initializer=self.init)(inputs)

        ### [First half of the network: downsampling inputs] ###
        lskip = []
        for lyn in lyns:
            skip = x
            x = self.add_DB(x, lyn=lyn, k=k)
            x = concatenate([skip, x])
            lskip.append(x)
            m += lyn*k
            x = self.add_TD(x, m=m)

        ### [Bottle neck] ###
        x = self.add_DB(x, lyn=15, k=k)
        m = 15*k

        ### [Second half of the network: upsampling inputs] ###
        lyns_rvs = lyns[::-1]
        lskip_rvs = lskip[::-1]
        for i, lyn in enumerate(lyns_rvs):
            x = self.add_TU(x, m=m)
            x = concatenate([lskip_rvs[i], x])
            x = self.add_DB(x, lyn=lyn, k=k)
            m = lyn*k


        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 1, activation="softmax", padding="same", kernel_initializer=self.init)(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model
