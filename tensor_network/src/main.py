import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

import src.tensor_network_layer as tnl

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def set_seed(seed: int = 200) -> None:
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_normal_network() -> tf.keras.Sequential:
    Dense = tf.keras.layers.Dense
    # 入力は2つのユニット、出力は1つのユニット
    # 全結合層は1024個のユニット 2x1024+1024=3x1024
    fc_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(1024, activation=tf.nn.relu),
            Dense(1024, activation=tf.nn.relu),
            Dense(1, activation=None),
        ]
    )
    return fc_model


def make_network_with_tensor_network() -> tf.keras.Sequential:
    Dense = tf.keras.layers.Dense
    tn_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(1024, activation=tf.nn.relu),
            # Here, we replace the dense layer with our MPS.
            tnl.TNLayer(),
            Dense(1, activation=None),
        ]
    )
    return tn_model


def make_training_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X: NDArray[np.float64] = np.concatenate(
        [
            np.random.randn(20, 2) + np.array([3, 3]),
            np.random.randn(20, 2) + np.array([-3, -3]),
            np.random.randn(20, 2) + np.array([-3, 3]),
            np.random.randn(20, 2) + np.array([3, -3]),
        ]
    )
    Y: NDArray[np.float64] = np.concatenate([np.ones((40)), -np.ones((40))])
    return X, Y


def make_test_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X: NDArray[np.float64] = np.concatenate(
        [
            np.random.randn(10, 2) + np.array([3, 3]),
            np.random.randn(10, 2) + np.array([-3, -3]),
            np.random.randn(10, 2) + np.array([-3, 3]),
            np.random.randn(10, 2) + np.array([3, -3]),
        ]
    )
    Y: NDArray[np.float64] = np.concatenate([np.ones((20)), -np.ones((20))])
    return X, Y


def make_grid_dataset(
    X: NDArray[np.float64], Y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    h = 1.0
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def predict(
    model: tf.keras.Sequential, xx: NDArray[np.float64], yy: NDArray[np.float64]
) -> NDArray[np.float64]:
    Z: NDArray[np.float64] = model.predict(np.c_[xx.ravel(), yy.ravel()])
    return Z


def plot_result(
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    filename: str,
) -> None:
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis("off")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.savefig(filename)


def calculate_accuracy(predicted_Y: NDArray[np.float64], test_Y: NDArray[np.float64]) -> float:
    s = 0
    for py, gy in zip(predicted_Y, test_Y):
        if py * gy > 0:
            s += 1
    return s / len(test_Y)


if __name__ == "__main__":
    set_seed()
    print("_/_/_/ make dataset")
    training_X, training_Y = make_training_dataset()
    # np.save("./dataset/training_X.npy", training_X)
    # np.save("./dataset/training_Y.npy", training_Y)

    NAME = "tensor_network"
    MAKE_NETWORK = {
        "normal": make_normal_network,
        "tensor_network": make_network_with_tensor_network,
    }

    print("_/_/_/ make model")
    model = MAKE_NETWORK[NAME]()
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    # model.fit(training_X, training_Y, epochs=300, verbose=0)

    # print("_/_/_/ make test dataset")
    # grid_X, grid_Y = make_grid_dataset(training_X, training_Y)

    # print("_/_/_/ predict on grid")
    # Z = model.predict(np.c_[grid_X.ravel(), grid_Y.ravel()])

    # print("_/_/_/ plot heatmap")
    # plot_result(grid_X, grid_Y, training_X, training_Y, Z, f"{NAME}.png")

    # print("_/_/_/ predict test dataset")
    # test_X, test_Y = make_test_dataset()
    # predicted_Y = model.predict(test_X)
    # accuracy = calculate_accuracy(predicted_Y, test_Y)
    # print(f"accuracy: {accuracy}")
