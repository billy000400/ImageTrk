from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# data_dir = Path("photographic_mc_FCDenseNet_channelByPlane2/photographic_large_train_X")
data_dir = Path("frcnn_mc_train_channelByPlane/mc_arrays_train")
file_list = sorted(data_dir.glob('*'))

print(file_list)
for file in file_list:
    print(str(file))
    input_photo = np.load(file)
    z, x, y = input_photo.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', alpha=1)
    plt.show()
    plt.close()
