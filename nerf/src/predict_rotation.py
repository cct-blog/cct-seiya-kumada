import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.camera as camera
import src.nerf as nerf
import src.nerf_loss as nloss


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir_path")
    parser.add_argument("--dataset_dir_name")
    parser.add_argument("--pose_dir_name")
    parser.add_argument("--image_ext")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--ind", type=int)
    parser.add_argument("--model_path")
    parser.add_argument("--t_n", type=float)
    parser.add_argument("--t_f", type=float)
    parser.add_argument("--output_path")
    parser.add_argument("--axis")
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    print(f"- dataset_dir_path: {args.dataset_dir_path}")
    print(f"- dataset_dir_name: {args.dataset_dir_name}")
    print(f"- pose_dir_name: {args.pose_dir_name}")
    print(f"- image_ext: {args.image_ext}")
    print(f"- width: {args.width}")
    print(f"- height: {args.height}")
    print(f"- model_path: {args.model_path}")
    print(f"- t_f: {args.t_n}")
    print(f"- t_f: {args.t_f}")
    print(f"- output_path: {args.output_path}")
    print(f"- ind: {args.ind}")
    assert args.axis == "z" or args.axis == "y"
    print(f"- axis: {args.axis}")


def reproduce_nerf(model_path: str, t_n: float, t_f: float) -> Any:
    checkpoint = torch.load(model_path)  # type:ignore

    model = nerf.NeRF(t_n=t_n, t_f=t_f, c_bg=(1, 1, 1))
    loss_fun = nloss.NeRFLoss(model)
    loss_fun.cuda("cuda:0")  # type:ignore

    loss_fun.load_state_dict(checkpoint["model_state_dict"])
    model = loss_fun.nerf
    return model


def rotate(model: Any, output_path: str, view: Any, axis: str) -> None:
    range = np.linspace(-np.pi, np.pi, 16)
    for i, a in enumerate(range):
        c = np.cos(a)
        s = np.sin(a)

        if axis == "y":
            # rotate around y-axis
            R = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32)
        else:
            # rotate around z-axis
            R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        _view = view.copy()
        _view["pose"] = R @ view["pose"]
        _, C_f = model(_view)
        plt.subplot(4, 4, i + 1)
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        plt.imshow(C_f)
    plt.savefig(output_path)


if __name__ == "__main__":
    args = extract_args()
    print_args(args)

    dataset_raw = camera.extract_outside_params(
        args.dataset_dir_path, args.dataset_dir_name, args.image_ext, args.pose_dir_name
    )

    pose = dataset_raw[args.ind]["pose"]
    rgb = dataset_raw[args.ind]["rgb"]

    camera_inside_params = camera.extract_inside_params(args.dataset_dir_path)
    camera.modify_inside_params(args.width, args.height, camera_inside_params)
    view = {
        "f": camera_inside_params.f,
        "cx": camera_inside_params.cy,
        "cy": camera_inside_params.cy,
        "origin_x": camera_inside_params.origin_x,
        "origin_y": camera_inside_params.origin_y,
        "origin_z": camera_inside_params.origin_z,
        "scale": camera_inside_params.scale,
        "height": int(camera_inside_params.img_height),
        "width": int(camera_inside_params.img_width),
        "pose": pose,
    }

    model = reproduce_nerf(args.model_path, args.t_n, args.t_f)
    rotate(model, args.output_path, view, args.axis)
