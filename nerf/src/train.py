import argparse
import os
import time
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

import src.camera as camera
import src.nerf as nerf
import src.nerf_loss as nloss


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir_path")
    parser.add_argument("--dataset_dir_name")
    parser.add_argument("--output_dir_path")
    parser.add_argument("--pose_dir_name")
    parser.add_argument("--image_ext")
    parser.add_argument("--saving_interval", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--t_n", type=float)
    parser.add_argument("--t_f", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resumes", action="store_true")
    parser.add_argument("--trained_model_path")
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    print(f"- dataset_dir_path: {args.dataset_dir_path}")
    print(f"- dataset_dir_name: {args.dataset_dir_name}")
    print(f"- output_dir_path: {args.output_dir_path}")
    print(f"- pose_dir_name: {args.pose_dir_name}")
    print(f"- image_ext: {args.image_ext}")
    print(f"- saving_interval: {args.saving_interval}")
    print(f"- width: {args.width}")
    print(f"- height: {args.height}")
    print(f"- t_n: {args.t_n}")
    print(f"- t_f: {args.t_f}")
    print(f"- epochs: {args.epochs}")
    print(f"- batch_size: {args.batch_size}")
    if args.resumes:
        print("- resumes!")
        print(f"- trained_model_path: {args.trained_model_path}")


def display_camera_params(camera_params: SimpleNamespace) -> None:
    print("Camera Params")
    print(f"> f: {camera_params.f}")
    print(f"> cx: {camera_params.cx}")
    print(f"> cy: {camera_params.cy}")
    print(f"> w: {camera_params.img_width}")
    print(f"> h: {camera_params.img_height}")


def save_state(loss_func: Any, optimizer: Any, e: int, dir_path: str) -> None:
    output_path = os.path.join(dir_path, f"epoch_{e}.state")
    torch.save(
        {
            "model_state_dict": loss_func.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_path,
    )


def train(
    dataset: Dict[str, Any],
    n_epoch: int,
    loss_func: Any,
    optimizer: Any,
    batch_size: int,
    saving_interval: int,
    output_dir_path: str,
) -> None:
    n_sample = dataset["o"].shape[0]
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)

    e: int = 0
    for e in range(1, n_epoch + 1):
        print("epoch: {}".format(e))
        perm = np.random.permutation(n_sample)
        sum_loss = 0.0
        sum_loss100 = 0.0

        for i in range(0, n_sample, batch_size):
            o = dataset["o"][perm[i : i + batch_size]]
            d = dataset["d"][perm[i : i + batch_size]]
            C = dataset["C"][perm[i : i + batch_size]]

            loss = loss_func(o, d, C)
            sum_loss += loss.item() * o.shape[0]
            sum_loss100 += loss.item()

            if (i / batch_size) % 100 == 99:
                print(f"{int(i/n_sample*100)}% {sum_loss100/100.}")
                sum_loss100 = 0.0

            loss_func.zero_grad()
            loss.backward()
            optimizer.step()

        print("sum loss: {}".format(sum_loss / n_sample))
        if e % saving_interval == 0:
            save_state(loss_func, optimizer, e, output_dir_path)

    # save a final state.
    save_state(loss_func, optimizer, e, output_dir_path)


def reproduce_model(trained_model_path: str, loss_fun: Any, opt: Any) -> None:
    checkpoint = torch.load(trained_model_path)  # type:ignore
    loss_fun.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])


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

    _dataset = nloss.preprocess_for_nerf_loss(dataset_raw, camera_inside_params)

    model = nerf.NeRF(t_n=args.t_n, t_f=args.t_f, c_bg=(1, 1, 1))
    loss_fun = nloss.NeRFLoss(model)
    loss_fun.cuda("cuda:0")  # type:ignore
    opt = torch.optim.Adam(loss_fun.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-7)
    if args.resumes:
        reproduce_model(args.trained_model_path, loss_fun, opt)

    dataset = {"o": _dataset["o"], "d": _dataset["d"], "C": _dataset["C"]}

    start = time.time()
    train(dataset, args.epochs, loss_fun, opt, args.batch_size, args.saving_interval, args.output_dir_path)
    end = time.time()
    print(f"{end - start}[sec]")
