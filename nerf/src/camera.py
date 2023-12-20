import glob
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def _line2floats(line: str) -> Any:
    return map(float, line.strip().split())


def extract_inside_params(dataset_path: str) -> SimpleNamespace:
    with open(os.path.join(dataset_path, "intrinsics.txt"), "r") as file:
        # focal length, image centers.
        f, cx, cy, _ = _line2floats(file.readline())

        # origin
        origin_x, origin_y, origin_z = _line2floats(file.readline())

        # near plane
        (near_plane,) = _line2floats(file.readline())

        # scale
        (scale,) = _line2floats(file.readline())

        # image size
        img_height, img_width = _line2floats(file.readline())

        return SimpleNamespace(
            f=f,
            cx=cx,
            cy=cy,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            near_plane=near_plane,
            scale=scale,
            img_height=img_height,
            img_width=img_width,
        )


def modify_inside_params(width: int, height: int, camera_params: SimpleNamespace) -> None:
    camera_params.f = camera_params.f * height / camera_params.img_height
    camera_params.cx = camera_params.cx * width / camera_params.img_width
    camera_params.cy = camera_params.cy * height / camera_params.img_height
    camera_params.img_width = width
    camera_params.img_height = height


# def extract_outside_params(
#    dataset_path: str, rgb_name: str, ext: str = "png"
# ) -> List[Dict[str, Any]]:
#    pose_paths = sorted(glob.glob(os.path.join(dataset_path, "pose/*.txt")))
#    rgb_paths = sorted(glob.glob(os.path.join(dataset_path, f"{rgb_name}/*.{ext}")))
#    dataset_raw = []
#
#    for pose_path, rgb_path in zip(pose_paths, rgb_paths):
#        pose = np.genfromtxt(pose_path, dtype=np.float32).reshape(4, 4)  # type:ignore
#
#        rgb = Image.open(rgb_path)
#
#        data = {
#            "pose": pose,
#            "rgb": rgb,
#        }
#        dataset_raw.append(data)
#    return dataset_raw


def extract_outside_params(dataset_path: str, rgb_name: str, ext: str = "png", pose_name: str = "pose") -> List[Dict[str, Any]]:
    pose_dir_path = os.path.join(dataset_path, pose_name)
    rgb_dir_path = os.path.join(dataset_path, f"{rgb_name}")

    pose_paths = sorted(glob.glob(os.path.join(pose_dir_path, "*.txt")))
    dataset_raw = []

    for pose_path in pose_paths:
        pose = np.genfromtxt(pose_path, dtype=np.float32).reshape(4, 4)  # type:ignore
        basename = os.path.basename(pose_path)
        head, _ = os.path.splitext(basename)
        rgb_path = os.path.join(rgb_dir_path, f"{head}.{ext}")
        rgb = Image.open(rgb_path)

        data = {
            "pose": pose,
            "rgb": rgb,
        }
        dataset_raw.append(data)
    return dataset_raw
