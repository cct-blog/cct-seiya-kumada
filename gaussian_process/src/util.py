from numpy.typing import NDArray
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

TRAIN_SIZE = 300
X_DIM = 10
DATA_PATH = "./data/data.txt"


def load_dataset(
    path: str, train_size: int
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    df = pd.read_table(path)
    ds = df.to_numpy()
    ys = ds[:, X_DIM]
    xs = ds[:, :X_DIM]
    xs = stats.zscore(xs)

    train_xs = xs[:train_size, :]
    test_xs = xs[train_size:, :]
    train_ys = ys[:train_size]
    test_ys = ys[train_size:]

    return train_xs, train_ys, test_xs, test_ys


def load_dataset_with_high_correlation(
    path: str, train_size: int
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    df = pd.read_table(path)
    ds = df.to_numpy()
    ys = ds[:, X_DIM]
    xs = ds[:, :X_DIM]
    rows, cols = xs.shape
    new_xs = np.ndarray((rows, 2), dtype=np.float32)
    new_xs[:, 0] = xs[:, 2]
    new_xs[:, 1] = xs[:, 8]
    # new_xs[:, 2] = xs[:, 3]
    # new_xs[:, 3] = xs[:, 7]
    new_xs = stats.zscore(new_xs)
    train_xs = new_xs[:train_size, :]
    test_xs = new_xs[train_size:, :]
    train_ys = ys[:train_size]
    test_ys = ys[train_size:]

    return train_xs, train_ys, test_xs, test_ys


def make_heat_map(data_path: str) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_table(data_path)
    ds = df.to_numpy()
    print(ds.shape)
    cs = np.corrcoef(ds.transpose())
    print(cs.shape)
    names = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]
    sns.heatmap(cs, annot=True, xticklabels=names, yticklabels=names)

    # グラフを表示する
    plt.savefig("heatmap.jpg")


if __name__ == "__main__":
    DATA_PATH = "./data/data.txt"
    # ヒートマップを作る。
    # make_heat_map(DATA_PATH)
    # Yと相関が強いのは BMIとS5の２つ。まずはこの２つでやってみる。
    train_xs, train_ys, test_xs, test_ys = load_dataset(DATA_PATH, 10)
    print(train_xs.dtype, train_ys.dtype)
