import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from numpy.typing import NDArray

import src.nerf as nerf
import src.nerf_loss as nloss


@dataclass
class RangeInfo:
    max: float = -sys.float_info.max
    min: float = sys.float_info.max


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--sigma_threshold", type=float)
    parser.add_argument("--size", type=int)
    parser.add_argument("--model_path")
    parser.add_argument("--t_n", type=float)
    parser.add_argument("--t_f", type=float)
    parser.add_argument("--ox", type=float)
    parser.add_argument("--oy", type=float)
    parser.add_argument("--oz", type=float)
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--zmin", type=float)
    parser.add_argument("--zmax", type=float)
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    print(f"- output_path: {args.output_path}")
    print(f"- sigma_threshold: {args.sigma_threshold}")
    print(f"- size: {args.size}")
    print(f"- model_path: {args.model_path}")
    print(f"- t_f: {args.t_n}")
    print(f"- t_f: {args.t_f}")
    print(f"- ox: {args.ox}")
    print(f"- oy: {args.oy}")
    print(f"- oz: {args.oz}")
    print(f"- xmin: {args.xmin}")
    print(f"- xmax: {args.xmax}")
    print(f"- ymin: {args.ymin}")
    print(f"- ymax: {args.ymax}")
    print(f"- zmin: {args.zmin}")
    print(f"- zmax: {args.zmax}")


def reproduce_nerf(model_path: str, t_n: float, t_f: float) -> Any:
    checkpoint = torch.load(model_path)  # type:ignore

    model = nerf.NeRF(t_n=t_n, t_f=t_f, c_bg=(1, 1, 1))
    loss_fun = nloss.NeRFLoss(model)
    loss_fun.cuda("cuda:0")  # type:ignore

    loss_fun.load_state_dict(checkpoint["model_state_dict"])
    model = loss_fun.nerf
    return model


def convert_to_uint8(v: float) -> int:
    v = int(v * 255)
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v


def judge_min_max(range: RangeInfo, v: float) -> None:
    if v > range.max:
        range.max = v
    if v < range.min:
        range.min = v


def save_points(
    ps: List[torch.Tensor],
    rgbs: NDArray[np.float32],
    sigmas: NDArray[np.float32],
    sigma_range: RangeInfo,
    x_range: RangeInfo,
    y_range: RangeInfo,
    z_range: RangeInfo,
    sigma_thr: float,
    fo: Any,
) -> None:
    for p, rgb, sigma in zip(ps, rgbs, sigmas):
        judge_min_max(sigma_range, sigma.item())
        if sigma.item() > sigma_thr:
            r, g, b = rgb
            if np.isnan(r):
                continue
            p = p.cpu().detach().numpy()[0, :]
            x, y, z = p
            judge_min_max(x_range, x)
            judge_min_max(y_range, y)
            judge_min_max(z_range, z)
            r = convert_to_uint8(r)
            g = convert_to_uint8(g)
            b = convert_to_uint8(b)
            fo.write(f"{x},{y},{z},{r},{g},{b}\n")


if __name__ == "__main__":
    args = extract_args()
    print_args(args)

    model = reproduce_nerf(args.model_path, args.t_n, args.t_f)
    fine_field = model.rf_f

    o = np.array([args.ox, args.oy, args.oz], dtype=np.float32)
    N = args.size
    xs = np.linspace(args.xmin, args.xmax, N)
    ys = np.linspace(args.ymin, args.ymax, N)
    zs = np.linspace(args.zmin, args.zmax, N)
    torch_o = torch.Tensor(o[np.newaxis, :]).cuda()
    sigma_threshold = args.sigma_threshold
    head, _ = os.path.split(args.output_path)
    if not os.path.isdir(head):
        os.makedirs(head)

    with open(args.output_path, "w") as fout:
        sigma_range = RangeInfo()
        x_range = RangeInfo()
        y_range = RangeInfo()
        z_range = RangeInfo()

        for index, z in enumerate(zs):
            print(f"> {index}/{N}")
            for y in ys:
                ps = []
                ds = []
                for x in xs:
                    p = torch.Tensor([x, y, z])
                    p = p.unsqueeze(dim=0).cuda()
                    d = p - torch_o
                    d = d / torch.norm(d)  # type:ignore
                    ps.append(p)
                    ds.append(d)
                torch_ps = torch.concat(ps, dim=0)
                torch_ds = torch.concat(ds, dim=0)
                torch_rgbs, torch_sigmas = fine_field(torch_ps, torch_ds)

                rgbs = torch_rgbs.cpu().detach().numpy()
                sigmas = torch_sigmas.cpu().detach().numpy()
                save_points(ps, rgbs, sigmas, sigma_range, x_range, y_range, z_range, sigma_threshold, fout)

        print(f"sigma range:{sigma_range.min},{sigma_range.max}")
        print(f"x range:{x_range.min},{x_range.max}")
        print(f"y range:{y_range.min},{y_range.max}")
        print(f"z range:{z_range.min},{z_range.max}")
