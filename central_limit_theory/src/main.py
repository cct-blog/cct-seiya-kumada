import matplotlib.pyplot as plt
import numpy as np


def calcuate_means(N: int, K: int) -> list[float]:
    x_means = []
    for _ in range(K):
        xs = []
        for n in range(N):
            x = np.random.rand()  # 一様分布からの乱数生成 [0,1)
            xs.append(x)
        mean = np.mean(xs)  # N個のサンプルの平均
        x_means.append(mean)  # K回繰り返す。
    return x_means


def draw_graph(x_means: list[float], N: int, K: int, path: str, mu: float, sigma: float) -> None:
    plt.hist(x_means, bins="auto", density=True)
    plt.title(f"N={N}, K={K}")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 5)
    # draw normal distribution
    x = np.linspace(-0.05, 1.05, 100)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    plt.plot(x, y, color="red", linestyle="dashed")
    plt.savefig(path)
    plt.clf()


def make_normal(mu: float, sigma: float):
    x = np.random.normal(mu, sigma)
    return x


if __name__ == "__main__":
    K = 10000
    MU = 0.5
    SIGMA = np.sqrt(1.0 / 12.0)
    for n in [1, 2, 4, 10]:
        x_means = calcuate_means(n, K)
        path = f"outputs/pdf_{n:02d}.png"
        draw_graph(x_means, n, K, path, MU, SIGMA / np.sqrt(n))
