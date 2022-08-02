from numpy.typing import NDArray
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

TRAIN_SIZE = 10
X_DIM = 10


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


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    DATA_PATH = "./data/data.txt"
    df = pd.read_table(DATA_PATH)
    ds = df.to_numpy()
    print(ds.shape)
    cs = np.corrcoef(ds.transpose())
    print(cs.shape)
    names = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]
    sns.heatmap(cs, annot=True, xticklabels=names, yticklabels=names)

    # グラフを表示する
    plt.savefig("heatmap.jpg")
    # Yと相関が強いのは BMIとS5の２つ。まずはこの２つでやってみる。
