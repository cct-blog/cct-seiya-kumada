import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import disk_level_set

OUTPUT_DIR_PATH = "init_level_set"
if __name__ == "__main__":
    image = np.zeros((143, 136))
    radius = [10, 20, 30, 40, 50, 60]

    # d_ls_imgs = []
    for i in radius:
        d_ls = disk_level_set(image.shape, radius=int(i))
        # d_ls_imgs.append(d_ls)
        path = os.path.join(OUTPUT_DIR_PATH, f"{i}.npy")
        np.save(path, d_ls)
        plt.imshow(d_ls)
        path = os.path.join(OUTPUT_DIR_PATH, f"{i}.jpg")
        plt.savefig(path)
        plt.clf()
