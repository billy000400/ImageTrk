from pathlib import Path

import numpy as np
import PIL.Image
from tensorflow.keras.utils import Sequence
import Config

class DataGenerator(Sequence):
    def __init__(self, X_dir, Y_dir, batch_size=1, shuffle=True):
        self.X_list = [child for child in X_dir.iterdir()]
        self.Y_list = [child for child in Y_dir.iterdir()]
        self.XY_list = list(zip(self.X_list, self.Y_list))
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.XY_list))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.XY_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        XY_file_list = [self.XY_list[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(XY_file_list)

        return X, Y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, XY_file_list):
        X_file_list = []
        Y_file_list = []
        for t in XY_file_list:
            X_file_list.append(t[0])
            Y_file_list.append(t[1])

        X = [np.load(x_file) for x_file in X_file_list]
        Y = [np.load(y_file) for y_file in Y_file_list]
        X = np.array(X, np.float32)
        Y = np.array(Y, np.float32)
        return X, Y

class ImageGenerator(Sequence):
    def __init__(self, X_dir, Y_dir, batch_size=1, shuffle=True):
        self.X_list = [child for child in X_dir.iterdir()]
        self.Y_list = [child for child in Y_dir.iterdir()]
        self.XY_list = list(zip(self.X_list, self.Y_list))
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.XY_list))
        self.shuffle = shuffle

        self.calc_normalization()
        self.on_epoch_end()


    def calc_normalization(self):
        X = []
        for file in self.X_list:
            img = PIL.Image.open(file)
            arr = np.array(img, dtype=np.float32)
            X.append(arr)
            img.close()
        X = np.array(X, dtype=np.float32)
        x_mean = X.mean()
        x_std = X.std()

        self.X_mean = x_mean
        self.X_std = x_std
        print('Normalization factor calculated')
        return

    def __len__(self):
        return int(np.floor(len(self.XY_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        XY_file_list = [self.XY_list[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(XY_file_list)
        return

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, XY_file_list):
        X_file_list = []
        Y_file_list = []
        for t in XY_file_list:
            X_file_list.append(t[0])
            Y_file_list.append(t[1])

        X = []
        for f in X_file_list:
            img = PIL.Image.open(f)
            img = img.resize((160,240))
            arr = np.array(img, dtype=np.float32)
            X.append(arr)
            img.close()

        Y = []
        for f in Y_file_list:
            print(f)
            img = PIL.Image.open(f)
            img = img.resize((160,240))
            arr = np.array(img, dtype=np.float32)
            Y.append(arr)
            img.close()

        X = np.array(X, np.float32)
        X = (X-self.X_mean)/self.X_std

        Y = np.array(Y, np.float32)
        Y[(Y==[255,255,255]).all(axis=3)] = np.array([1,0,0], dtype=np.float32)
        Y[(Y==[0,0,255]).all(axis=3)] = np.array([0,1,0], dtype=np.float32)
        Y[(Y==[255,0,0]).all(axis=3)] = np.array([0,0,1], dtype=np.float32)

        return X, Y
