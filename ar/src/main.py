from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass
class Ar1Process:
    c: float
    phi: float
    sigma: float

    def __init__(self, c: float, phi: float, sigma: float):
        self.c = c
        self.phi = phi
        self.sigma = sigma
        self.values = [c]

    def __call__(self, x: float) -> float:
        e: float = np.random.normal(0, self.sigma, 1)[0]
        pre_value: float = self.values[-1]
        cur_value = self.c + self.phi * pre_value + e
        self.values.append(cur_value)
        return cur_value


def draw_graph(xs: NDArray[np.float64], ys: List[float], filename: str) -> None:
    plt.plot(xs, ys)
    plt.xlabel("t")
    # plt.ylim(-5, 5)
    plt.ylabel("y")
    # plt.legend(loc="best")
    plt.savefig(f"./images/{filename}")
    plt.clf()


if __name__ == "__main__":
    xs = np.linspace(start=0, stop=100, num=100)

    c = 0.0
    phi = 1.1
    sigma = 1.0
    ar1_process = Ar1Process(c=c, phi=phi, sigma=sigma)

    ys = [ar1_process(x) for x in xs]
    # print(type(xs), type(ys))
    draw_graph(xs, ys, f"ar1_c_{c}_phi_{phi}_sigma_{sigma}.jpg")
