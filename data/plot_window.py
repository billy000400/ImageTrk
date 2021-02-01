import sys
from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

cwd = Path.cwd()
bbox_file = cwd.joinpath('bbox_proposal_train.csv')
img_dir = cwd.joinpath('imgs_train')

df = pd.read_csv(bbox_file)

img_files = [child for child in img_dir.iterdir()]

for img_file in img_files:
    img_name = img_file.name
    bboxes = df[df['FileName']==img_name]

    img = mpimg.imread(img_file)
    height = img.shape[0]
    width = img.shape[1]

    imgplot = plt.imshow(img, interpolation='none')

    # for index, bbox in bboxes.iterrows():
    #     xmin = bbox['XMin']
    #     xmax = bbox['XMax']
    #     ymin = bbox['YMin']
    #     ymax = bbox['YMax']
    #
    #     xmin = xmin*height
    #     xmax = xmax*width
    #     ymin, ymax = (1-ymax)*800, (1-ymin)*800
    #
    #
    #     rec_xy = (xmin, ymin)
    #     rec_width = abs(xmax-xmin)
    #     rec_height = abs(ymax-ymin)
    #     rect=Rectangle(rec_xy,rec_width,rec_height,linewidth=1,edgecolor='r',facecolor='none')
    #     plt.gca().add_patch(rect)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('The XY Projection of StrawDigiMCs')
    plt.show()
    sys.exit()
