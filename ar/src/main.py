from dataclasses import dataclass

import numpy as np


@dataclass
class Ar1Process:
    c: float
    phi: float
    sigma: float


if __name__ == "__main__":
    xs = np.linspace(start=0, stop=100, num=100)

    c = 0.0
    phi = 0.8
    sigma = 1.0
    process = Ar1Process(c=c, phi=phi, sigma=sigma)
    print("hello")
