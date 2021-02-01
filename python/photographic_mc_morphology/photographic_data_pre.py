import sys
import pickle
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Config import extractor_config as Config
from mu2e_output import *

pinfo('Loading Config')
cwd = Path.cwd()
pickle_path = cwd.joinpath('photographic.train.config.pickle')
C = pickle.load(open(pickle_path,'rb'))

pinfo('Making directories')
data_dir = Path.cwd().parent.parent.joinpath('data')
morph_dir = data_dir.joinpath('photographic_morphology')
X_train_dir = morph_dir.joinpath('Arc_train').joinpath('X')
Y_train_dir = morph_dir.joinpath('Arc_train').joinpath('Y')
X_val_dir = morph_dir.joinpath('Arc_val').joinpath('X')
Y_val_dir = morph_dir.joinpath('Arc_val').joinpath('Y')

train_X_list = [child for child in X_train_dir.iterdir()]
train_Y_list = [child for child in Y_train_dir.iterdir()]

val_X_list = [child for child in X_val_dir.iterdir()]
val_Y_list = [child for child in Y_val_dir.iterdir()]

to_X_train_dir = morph_dir.joinpath('X_train_npy')
to_Y_train_dir = morph_dir.joinpath('Y_train_npy')

to_X_val_dir = morph_dir.joinpath('X_val_npy')
to_Y_val_dir = morph_dir.joinpath('Y_val_npy')

new_dirs = [to_X_train_dir, to_Y_train_dir, to_X_val_dir, to_Y_val_dir]
for dir in new_dirs:
    shutil.rmtree(dir, ignore_errors=True)
    dir.mkdir(parents=True, exist_ok=True)

pinfo('Calculating Normalization Parameters')
res = C.resolution
shape = (res, int(res/3*2)) # This is width, height for Image
X_train = []
for f in train_X_list:
    img=Image.open(f)
    img=img.resize(shape)
    arr=np.array(img, dtype=np.float32)
    X_train.append(arr)
    img.close()
X_train = np.array(X_train, dtype=np.float32)
X_mean = X_train.mean()
X_std = X_train.std()

pinfo('Converting X train data')
X_train = (X_train-X_mean)/X_std
for i, f in enumerate(train_X_list):
    name = f.stem
    to_file = to_X_train_dir.joinpath(name)
    np.save(to_file, X_train[i])

pinfo('Converting X validation data')
for f in val_X_list:
    img=Image.open(f)
    img=img.resize(shape)
    arr=np.array(img, dtype=np.float32)
    img.close()
    arr=(arr-X_mean)/X_std

    name = f.stem
    to_file = to_X_val_dir.joinpath(name)
    np.save(to_file, arr)

pinfo('Converting Y train data')
is_blank =  np.array([1,0,0], dtype=np.float32)
is_bg = np.array([0,1,0], dtype=np.float32)
is_major = np.array([0,0,1], dtype=np.float32)
for f in train_Y_list:
    img=Image.open(f)
    img=img.resize(shape)
    y=np.array(img, dtype=np.float32)
    img.close()

    y[(y==[255,255,255]).all(axis=2)] = is_blank
    y[(y==[0,0,255]).all(axis=2)] = is_bg
    y[(y==[255,0,0]).all(axis=2)] = is_major
    print(y)
    name=f.stem
    to_file = to_Y_train_dir.joinpath(name)
    np.save(to_file,y)

pinfo('Converting Y validation data')
for f in val_Y_list:
    img=Image.open(f)
    img=img.resize(shape)
    y=np.array(img, dtype=np.float32)
    img.close()

    y[(y==[255,255,255]).all(axis=2)] = is_blank
    y[(y==[0,0,255]).all(axis=2)] = is_bg
    y[(y==[255,0,0]).all(axis=2)] = is_major

    name=f.stem
    to_file = to_Y_val_dir.joinpath(name)
    np.save(to_file,y)

# load pickle
pinfo('Setting training and validation directories')


C.set_train_dir(to_X_train_dir, to_Y_train_dir)
C.set_val_dir(to_X_val_dir, to_Y_val_dir)
pickle.dump(C, open(pickle_path, 'wb'))
