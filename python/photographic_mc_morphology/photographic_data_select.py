from pathlib import Path
import shutil
import csv

data_dir = Path.cwd().parent.parent.joinpath('data')
X_train_dir = data_dir.joinpath('photographic_morphology_train_X')
Y_train_dir = data_dir.joinpath('photographic_morphology_train_Y')
X_val_dir = data_dir.joinpath('photographic_morphology_val_X')
Y_val_dir = data_dir.joinpath('photographic_morphology_val_Y')
morph_dir = data_dir.joinpath('photographic_morphology')

spiral_train_file = Path.cwd().joinpath('spiral_train0.txt')
spiral_val_file = Path.cwd().joinpath('spiral_validation0.txt')

arc_train_dir = morph_dir.joinpath('Arc_train')
arc_val_dir = morph_dir.joinpath('Arc_val')

spiral_train_dir = morph_dir.joinpath('Spiral_train')
spiral_val_dir = morph_dir.joinpath('Spiral_val')

### make new dirs
dir_list = [arc_train_dir, arc_val_dir, spiral_train_dir, spiral_val_dir]
for dir in dir_list:
    shutil.rmtree(dir, ignore_errors=True)
    dir.mkdir(parents=True, exist_ok=True)
    x_dir = dir.joinpath('X')
    y_dir = dir.joinpath('Y')
    x_dir.mkdir(parents=True, exist_ok=True)
    y_dir.mkdir(parents=True, exist_ok=True)

### get the sub list of spiral training data
train_X_list_spiral = []
train_Y_list_spiral = []
with open(spiral_train_file) as f:
    reader = csv.reader(f)
    for row in reader:
        index = row[0]
        train_X_list_spiral.append(X_train_dir.joinpath('input_'+index.zfill(6)+'.npy'))
        train_Y_list_spiral.append(Y_train_dir.joinpath('output_'+index.zfill(6)+'.npy'))

### get the sub list of spiral validation data
val_X_list_spiral = []
val_Y_list_spiral = []
with open(spiral_val_file) as f:
    reader = csv.reader(f)
    for row in reader:
        index = row[0]
        val_X_list_spiral.append(X_val_dir.joinpath('input_'+index.zfill(6)+'.npy'))
        val_Y_list_spiral.append(Y_val_dir.joinpath('output_'+index.zfill(6)+'.npy'))

### split X_train data into arc_train and spiral_train
X_train_files = [child for child in X_train_dir.iterdir()]
for file in X_train_files:
    if file in train_X_list_spiral:
        name = file.name
        to_file = spiral_train_dir.joinpath('X').joinpath(name)
        shutil.copy(file,to_file)
    else:
        name = file.name
        to_file = arc_train_dir.joinpath('X').joinpath(name)
        shutil.copy(file,to_file)

### split Y_train data into arc_train and spiral_train
Y_train_files = [child for child in Y_train_dir.iterdir()]
for file in Y_train_files:
    if file in train_Y_list_spiral:
        name = file.name
        to_file = spiral_train_dir.joinpath('Y').joinpath(name)
        shutil.copy(file,to_file)
    else:
        name = file.name
        to_file = arc_train_dir.joinpath('Y').joinpath(name)
        shutil.copy(file,to_file)

### split X_val into arc_val and spiral_val
X_val_files = [child for child in X_val_dir.iterdir()]
for file in X_val_files:
    if file in val_X_list_spiral:
        name = file.name
        to_file = spiral_val_dir.joinpath('X').joinpath(name)
        shutil.copy(file,to_file)
    else:
        name = file.name
        to_file = arc_val_dir.joinpath('X').joinpath(name)
        shutil.copy(file,to_file)

### split Y_val into arc_val and spiral_val
Y_val_files = [child for child in Y_val_dir.iterdir()]
for file in Y_val_files:
    if file in val_Y_list_spiral:
        name = file.name
        to_file = spiral_val_dir.joinpath('Y').joinpath(name)
        shutil.copy(file,to_file)
    else:
        name = file.name
        to_file = arc_val_dir.joinpath('Y').joinpath(name)
        shutil.copy(file,to_file)
