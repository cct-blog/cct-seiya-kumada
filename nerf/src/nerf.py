from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from src.radiance_fields import RadianceField
from src.rendering import volume_rendering_with_radiance_field


def camera_params_to_rays(
    f: float,
    cx: float,
    cy: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    scale: float,
    pose: NDArray[np.float32],
    width: int,
    height: int,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Make rays (o, d) from camera parameters.

    Args:
        f (float): A focal length.
        cx, cy (float): A center of the image.
        s (float): scale.
        pose (ndarray, [4, 4]): camera extrinsic matrix.
        width(int): The height of the rendered image.
        height(int): The width of the rendered image.

    Returns:
        o (ndarray, [height, width, 3]): The origin of the camera coordinate.
        d (ndarray, [height, width, 3]): The direction of each ray.
        These values are described in the world coordinate system.
    """
    _o = np.zeros((height, width, 4), dtype=np.float32)
    _o[:, :, 0] = origin_x
    _o[:, :, 1] = origin_y
    _o[:, :, 2] = origin_z
    _o[:, :, 3] = 1

    # coordinates of all pixels
    v, u = np.mgrid[:height, :width].astype(np.float32)
    _x = scale * (u - cx) / f
    _y = scale * (v - cy) / f
    _z = scale * np.ones_like(_x)
    _w = np.ones_like(_x)
    _d = np.stack([_x, _y, _z, _w], axis=2)

    # transform the camera coordinate into the world coordinate
    o = (pose @ _o[..., None])[..., :3, 0]
    _d = (pose @ _d[..., None])[..., :3, 0]
    d = _d - o
    d /= np.linalg.norm(d, axis=2, keepdims=True)  # type:ignore
    # "o" and "d" are described using the world coordinate system.
    # o.shape==(height,width,3)
    # d.shape==(height,width,3)
    return o, d


class NeRF(nn.Module):

    # sampling parameter
    N_c = 64
    N_f = 128

    # batchsize
    N_SAMPLES = 2048

    def __init__(
        self,
        t_n: float = 0.0,
        t_f: float = 2.5,
        L_x: int = 10,
        L_d: int = 4,
        c_bg: Tuple[int, int, int] = (1, 1, 1),
    ):
        self.t_n = t_n
        self.t_f = t_f
        self.c_bg = c_bg

        super(NeRF, self).__init__()
        self.rf_c = RadianceField(L_x=L_x, L_d=L_d)
        self.rf_f = RadianceField(L_x=L_x, L_d=L_d)

    def device(self) -> torch.device:
        return next(self.parameters()).device

    # used for prediction
    def forward(self, view: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render Image with view paramters.

        Args:
            view (dict): View (camera) parameters.
                view = {
                    # intrinsic paramters.
                    f: <float, the focal length.>,
                    cx : <float, the center of the image (x).>,
                    cy : <float, the center of the image (y).>,
                    width: <int, the image width.>,
                    height: <int, the image height.>,
                    # extrinsic parameter.
                    pose: <ndarray, [4, 4], camera extrinsic matrix.>
                }

        Returns:
            C_c (ndarray, [height, width, 3]): The rendered image (coarse).
            C_f (ndarray, [height, width, 3]): The rendered image (fine).

        """
        f = view["f"]
        cx = view["cx"]
        cy = view["cy"]
        origin_x = view["origin_x"]
        origin_y = view["origin_y"]
        origin_z = view["origin_z"]
        scale = view["scale"]
        pose = view["pose"]
        width = view["width"]
        height = view["height"]

        o, d = camera_params_to_rays(f, cx, cy, origin_x, origin_y, origin_z, scale, pose, width, height)
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)

        device = self.device()
        o = torch.tensor(o, device=device)  # type:ignore
        d = torch.tensor(d, device=device)  # type:ignore

        _C_c = []
        _C_f = []
        with torch.no_grad():
            for i in range(0, o.shape[0], self.N_SAMPLES):
                o_i = o[i : i + self.N_SAMPLES]
                d_i = d[i : i + self.N_SAMPLES]
                C_c_i, C_f_i = volume_rendering_with_radiance_field(
                    self.rf_c,
                    self.rf_f,
                    o_i,
                    d_i,
                    self.t_n,
                    self.t_f,
                    N_c=self.N_c,
                    N_f=self.N_f,
                    c_bg=self.c_bg,
                )
                _C_c.append(C_c_i.cpu().numpy())
                _C_f.append(C_f_i.cpu().numpy())

        C_c = np.concatenate(_C_c, axis=0)  # type:ignore
        C_f = np.concatenate(_C_f, axis=0)  # type:ignore
        C_c = np.clip(0.0, 1.0, C_c.reshape(height, width, 3))
        C_f = np.clip(0.0, 1.0, C_f.reshape(height, width, 3))

        return C_c, C_f
