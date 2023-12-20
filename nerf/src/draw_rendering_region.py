import argparse
import os
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import src.camera as camera
import src.nerf as nerf


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir_path")
    parser.add_argument("--dataset_dir_name")
    parser.add_argument("--output_path")
    parser.add_argument("--pose_dir_name")
    parser.add_argument("--image_ext")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--t_n", type=float)
    parser.add_argument("--t_f", type=float)
    parser.add_argument("--elev", type=int)
    parser.add_argument("--azim", type=int)
    parser.add_argument("--camera_interval", type=int)
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    print(f"- dataset_dir_path: {args.dataset_dir_path}")
    print(f"- dataset_dir_name: {args.dataset_dir_name}")
    print(f"- output_dir_path: {args.output_path}")
    print(f"- pose_dir_name: {args.pose_dir_name}")
    print(f"- image_ext: {args.image_ext}")
    print(f"- width: {args.width}")
    print(f"- height: {args.height}")
    print(f"- t_n: {args.t_n}")
    print(f"- t_f: {args.t_f}")
    print(f"- elev: {args.elev}")
    print(f"- azim: {args.azim}")
    print(f"- camera_interval: {args.camera_interval}")


def draw_rendering_region(
    dataset_raw: Any, params: SimpleNamespace, t_n: float, t_f: float, path: str, elev: int, azim: int, camera_interval: int
) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(-2, 2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    focuses = []

    for data in dataset_raw[::camera_interval]:
        o, d = nerf.camera_params_to_rays(
            params.f,
            params.cx,
            params.cy,
            params.origin_x,
            params.origin_y,
            params.origin_z,
            params.scale,
            data["pose"],
            params.img_width,
            params.img_height,
        )

        # 焦点（赤）
        o_x, o_y, o_z = o[0, :1].T
        # if np.abs(o_z) > 0.5:
        #    continue
        ax.scatter(o_x, o_y, o_z, c="red")
        focuses.append(np.array([o_x, o_y, o_z]))

        # レンダリング下限（青）
        N = 5  # interval
        x_n, y_n, z_n = (o + d * t_n)[::N, ::N].reshape(-1, 3).T
        ax.scatter(x_n, y_n, z_n, c="blue", s=0.1)

        # レンダリング上限（緑）
        x_f, y_f, z_f = (o + d * t_f)[::N, ::N].reshape(-1, 3).T
        ax.scatter(x_f, y_f, z_f, c="green", s=0.1)
    if len(focuses) == 0:
        print("ERROR: focuses not found!")
        return
    dir_path, tail = os.path.split(path)
    name, ext = os.path.splitext(tail)
    new_name = f"{name}_{elev}_{azim}{ext}"
    new_path = os.path.join(dir_path, new_name)
    plt.savefig(new_path)
    focuses_ = np.array(focuses).squeeze()
    print("average focus: ", np.average(focuses_, axis=0))  # type:ignore


if __name__ == "__main__":
    args = extract_args()
    print_args(args)

    # extract inside camera parameters
    camera_inside_params = camera.extract_inside_params(args.dataset_dir_path)

    # modify inside camera parameters
    camera.modify_inside_params(args.width, args.height, camera_inside_params)

    # extract outside camera parameters
    dataset_raw = camera.extract_outside_params(
        args.dataset_dir_path, args.dataset_dir_name, args.image_ext, args.pose_dir_name
    )

    head, _ = os.path.split(args.output_path)
    if not os.path.isdir(head):
        os.makedirs(head)

    draw_rendering_region(
        dataset_raw, camera_inside_params, args.t_n, args.t_f, args.output_path, args.elev, args.azim, args.camera_interval
    )
