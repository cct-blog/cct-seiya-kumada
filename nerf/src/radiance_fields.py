from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gamma(p: torch.Tensor, L: int) -> torch.Tensor:
    """Encode positions.

    Args:
        p (ndarray, [batch_size, dim]): Position.
        L (int): encoding param.

    Returns:
        ndarray [batch_size, dim * L]: Encoded position.

    """
    # normalization.
    p = torch.tanh(p)

    batch_size = p.shape[0]
    i = torch.arange(L, dtype=torch.float32, device=p.device)
    a = (2.0 ** i[None, None]) * np.pi * p[:, :, None]
    s = torch.sin(a)
    c = torch.cos(a)
    e: torch.Tensor = torch.cat([s, c], axis=2).view(batch_size, -1)  # type:ignore
    return e


def _init_weights(m: Any) -> None:
    if type(m) is nn.Linear:
        nn.init.kaiming_normal_(m.weight)  # type:ignore
        nn.init.zeros_(m.bias)


class RadianceField(nn.Module):
    """Radiance Field Functions.

    This is ``$F_\theta$`` in the paper.

    """

    def __init__(self, L_x: int = 10, L_d: int = 4) -> None:
        # positional encoding parameter.
        self.L_x = L_x
        self.L_d = L_d

        super(RadianceField, self).__init__()
        self.layer0 = nn.Linear(6 * L_x, 256)
        self.layer1 = nn.Linear(256, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256 + 6 * L_x, 256)
        self.layer6 = nn.Linear(256, 256)
        self.layer7 = nn.Linear(256, 256)
        self.sigma = nn.Linear(256, 1)
        self.layer8 = nn.Linear(256, 256)
        self.layer9 = nn.Linear(256 + 6 * L_d, 128)
        self.layer10 = nn.Linear(128, 128)
        self.layer11 = nn.Linear(128, 128)
        self.layer12 = nn.Linear(128, 128)
        self.rgb = nn.Linear(128, 3)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply function.

        Args:
            x (tensor, [batch_size, 3]): Points on rays.
            d (tensor, [batch_size, 3]): Direction of rays.

        Returns:
            rgb (tensor, [batch_size, 3]): Emitted color.
            sigma (tensor, [batch_size, 1]): Volume density.

        """
        # positional encoding.
        e_x = gamma(x, self.L_x)
        e_d = gamma(d, self.L_d)

        # forward
        h = F.relu(self.layer0(e_x))
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(h))
        h = F.relu(self.layer4(h))
        h = torch.cat([h, e_x], axis=1)  # type:ignore
        h = F.relu(self.layer5(h))
        h = F.relu(self.layer6(h))
        h = F.relu(self.layer7(h))
        sigma = F.relu(self.sigma(h))
        h = self.layer8(h)
        h = torch.cat([h, e_d], axis=1)  # type:ignore
        h = F.relu(self.layer9(h))
        h = F.relu(self.layer10(h))
        h = F.relu(self.layer11(h))
        h = F.relu(self.layer12(h))
        rgb = torch.sigmoid(self.rgb(h))

        return rgb, sigma
