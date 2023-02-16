from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

np.random.seed(1)


@dataclass
class Ma1Process:
    theta: float
    mu: float
    sigma: float  # 標準偏差
    noises: List[float]

    def __init__(self, theta: float, mu: float, sigma: float):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.noises = [0.0]

    def __call__(self, x: float) -> float:
        cur_e: float = np.random.normal(0, self.sigma, 1)[0]
        pre_e: float = self.noises[-1]
        self.noises.append(cur_e)
        return self.mu + cur_e + self.theta * pre_e


def draw_graph(xs: NDArray[np.float64], ys: List[float], filename: str) -> None:
    plt.plot(xs, ys)
    plt.xlabel("t")
    plt.ylim(-5, 5)
    plt.ylabel("y")
    # plt.legend(loc="best")
    plt.savefig(f"./images/{filename}")
    plt.clf()


if __name__ == "__main__":
    xs = np.linspace(start=0, stop=100, num=100)

    mu = 0.0
    theta = -0.1
    sigma = 1.0
    ma1_process = Ma1Process(mu=mu, theta=theta, sigma=sigma)
    ys = [ma1_process(x) for x in xs]
    # print(type(xs), type(ys))
    draw_graph(xs, ys, f"ma1_mu_{mu}_theta_{theta}_sigma_{sigma}.jpg")
