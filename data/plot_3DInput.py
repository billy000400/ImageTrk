from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

data_dir = Path("photographic_mc_FCDenseNet_channelByPlane2/photographic_large_train_X")
file_list = sorted(data_dir.glob('*'))

for file in file_list:
    input_photo = np.load(file)
    z, x, y = input_photo.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
    plt.close()
