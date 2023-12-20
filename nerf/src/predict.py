import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
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
    parser.add_argument("--view_path")
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
    print(f"- view_path: {args.view_path}")
    print(f"- ind: {args.ind}")


def save_view(C_c: torch.Tensor, C_f: torch.Tensor, rgb: Any, view_path: str) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("coarse")
    plt.imshow(C_c)

    plt.subplot(1, 3, 2)
    plt.title("fine")
    plt.imshow(C_f)

    plt.subplot(1, 3, 3)
    plt.title("original")
    plt.imshow(rgb)

    head, _ = os.path.split(view_path)
    if not os.path.isdir(head):
        os.makedirs(head)

    plt.savefig(view_path)


def reproduce_nerf(model_path: str, t_n: float, t_f: float) -> Any:
    checkpoint = torch.load(model_path)  # type:ignore

    model = nerf.NeRF(t_n=t_n, t_f=t_f, c_bg=(1, 1, 1))
    loss_fun = nloss.NeRFLoss(model)
    loss_fun.cuda("cuda:0")  # type:ignore

    loss_fun.load_state_dict(checkpoint["model_state_dict"])
    model = loss_fun.nerf
    return model


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
        "cx": camera_inside_params.cx,
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

    C_c, C_f = model(view)
    save_view(C_c, C_f, rgb, args.view_path)
