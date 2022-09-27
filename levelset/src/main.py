import os
from typing import Any

import cv2
import numpy as np
import skimage.segmentation as seg


def convert_to_gray(init: Any) -> Any:
    a = init.astype(np.uint32)
    a = np.where(a == 1, 255, 0)
    b = a.astype(np.uint8)
    return b


# https://sabopy.com/py/scikit-image-66/
IMAGE_PATH = "C:\\data\\levelset\\apple_2.png"
IMAGE_PATH_2 = "init_level_set\\10.npy"
if __name__ == "__main__":

    image = cv2.imread(IMAGE_PATH, 0)
    init = np.load(IMAGE_PATH_2)
    _, tail = os.path.split(IMAGE_PATH)
    head, _ = os.path.splitext(tail)
    dir_path = os.path.join("outputs", head)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    for iter in range(0, 100, 5):
        print(f"iter: {iter}")
        cv = seg.chan_vese(
            image, mu=0.2, lambda1=1, lambda2=1, tol=1e-3, max_iter=iter, dt=0.5, init_level_set=init, extended_output=True,
        )

        dst = cv[0]
        dst = convert_to_gray(dst)
        dst = cv2.Canny(dst, 100, 200)
        path = os.path.join(dir_path, f"{iter}.jpg")
        cv2.imwrite(path, dst)
