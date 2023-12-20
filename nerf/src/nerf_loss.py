from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.nerf as nerf
from src.nerf import NeRF
from src.rendering import volume_rendering_with_radiance_field

# from numpy.typing import NDArray


class NeRFLoss(nn.Module):
    def __init__(self, model: NeRF):
        super(NeRFLoss, self).__init__()
        self.nerf = model

    def forward(self, o: torch.Tensor, d: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        device = self.nerf.device()
        o = torch.tensor(o, device=device)
        d = torch.tensor(d, device=device)
        C = torch.tensor(C, device=device)

        rf_c = self.nerf.rf_c
        rf_f = self.nerf.rf_f
        t_n = self.nerf.t_n
        t_f = self.nerf.t_f
        N_c = self.nerf.N_c
        N_f = self.nerf.N_f
        c_bg = self.nerf.c_bg
        C_c, C_f = volume_rendering_with_radiance_field(rf_c, rf_f, o, d, t_n, t_f, N_c=N_c, N_f=N_f, c_bg=c_bg)

        loss = F.mse_loss(C_c, C) + F.mse_loss(C_f, C)
        return loss


def preprocess_for_nerf_loss(dataset_raw: List[Dict[str, Any]], inside_params: SimpleNamespace) -> Dict[str, Any]:

    os = []
    ds = []
    Cs = []

    for data in dataset_raw:
        pose = data["pose"]
        rgb = data["rgb"]

        o, d = nerf.camera_params_to_rays(
            inside_params.f,
            inside_params.cx,
            inside_params.cy,
            inside_params.origin_x,
            inside_params.origin_y,
            inside_params.origin_z,
            inside_params.scale,
            pose,
            inside_params.img_width,
            inside_params.img_height,
        )
        C = (np.array(rgb, dtype=np.float32) / 255.0)[:, :, :3]  # type:ignore

        # 3 means (x,y,z).
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        # 3 means (R,G,B).
        C = C.reshape(-1, 3)

        os.append(o)
        ds.append(d)
        Cs.append(C)

    os = np.concatenate(os)  # type:ignore
    ds = np.concatenate(ds)  # type:ignore
    Cs = np.concatenate(Cs)  # type:ignore

    dataset = {"o": os, "d": ds, "C": Cs}
    return dataset
    # 保存しておく
    # np.savez(output_path, **dataset)  # type:ignore
