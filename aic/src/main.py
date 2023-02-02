from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1)


def polynomial(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2.0 * xs - np.sin(3.0 * np.pi * xs)  # type:ignore


def make_dataset(n: int, scale: float) -> List[NDArray[np.float64]]:
    xs = np.linspace(start=0, stop=1, num=50)
    ys = polynomial(xs)
    random_xs = np.random.rand(n)
    random_ys = polynomial(random_xs) + np.random.normal(
        loc=0, scale=scale, size=n
    )
    return [xs, ys, random_xs, random_ys]


def draw_figure(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
) -> None:
    plt.plot(xs, ys, label="ground truth")
    plt.scatter(random_xs, random_ys, color="red", label="observed")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(-2, 3)
    plt.legend(loc="best")
    plt.savefig("./images/data.jpg")
    plt.clf()


def draw_figure_with_prediction(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
    pred_ys: NDArray[np.float64],
) -> None:
    plt.plot(xs, ys, label="ground truth")
    plt.plot(xs, pred_ys, label="prediction")
    plt.scatter(random_xs, random_ys, color="red", label="observed")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(-2, 3)
    plt.legend(loc="best")
    plt.savefig("./images/data_with_prediction.jpg")
    plt.clf()


# https://cpp-learning.com/polynomial-regression/#sklearn2
def execute_regression(
    xs: NDArray[np.float64], ys: NDArray[np.float64], degree: int
) -> Tuple[LinearRegression, PolynomialFeatures]:
    # 次元を追加（ライブラリの仕様に合わせる）
    xs = xs[:, np.newaxis]
    ys = ys[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=degree)
    xs_poly = polynomial_features.fit_transform(xs)
    model = LinearRegression()
    model.fit(xs_poly, ys)
    return model, polynomial_features


def predict(
    model: LinearRegression,
    pf: PolynomialFeatures,
    xs: NDArray[np.float64],
) -> NDArray[np.float64]:
    xs = xs[:, np.newaxis]
    xs = pf.fit_transform(xs)
    ys = model.predict(xs)
    return ys  # type:ignore


def calculate_mse(
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
    model: LinearRegression,
    pf: PolynomialFeatures,
) -> np.float64:
    pred_ys = predict(model, pf, random_xs)
    rmse = mean_squared_error(random_ys, pred_ys)
    return rmse  # type:ignore


def calculate_aic_and_bic(
    mse: np.float64, degree: int, data_num: int
) -> Tuple[np.float64, np.float64]:
    a = data_num * mse
    return (a + 2 * degree, a + degree * np.log(data_num))


def draw_aic_and_bic(
    degrees: List[int], aics: List[np.float64], bics: List[np.float64]
) -> None:
    plt.plot(degrees, aics, label="AIC", color="red")
    plt.scatter(degrees, aics, color="red")
    plt.plot(degrees, bics, label="BIC", color="blue")
    plt.scatter(degrees, bics, color="blue")
    plt.xlabel("Degree")
    plt.xticks(degrees)
    plt.ylabel("AIC/BIC")
    plt.legend(loc="best")
    plt.savefig("./images/aic_bic.jpg")
    plt.clf()


def make_aic_and_bic(
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
    N: int,
    max_degree: int,
) -> None:
    aics = []
    bics = []
    degrees = list(range(2, max_degree))
    for degree in degrees:
        # 多項式で回帰する。
        model, polynomial_features = execute_regression(
            random_xs, random_ys, degree
        )

        # 評価する。
        mse = calculate_mse(random_xs, random_ys, model, polynomial_features)

        (aic, bic) = calculate_aic_and_bic(mse, degree, N)
        aics.append(aic)
        bics.append(bic)

    draw_aic_and_bic(degrees, aics, bics)


def execute_polynomial_regression(
    degree: int,
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    random_xs: NDArray[np.float64],
    random_ys: NDArray[np.float64],
) -> None:
    model, polynomial_features = execute_regression(
        random_xs, random_ys, degree
    )

    # 予測する。
    pred_ys = predict(model, polynomial_features, xs)

    # 可視化する。
    draw_figure_with_prediction(xs, ys, random_xs, random_ys, pred_ys)

    # 評価する。
    mse = calculate_mse(random_xs, random_ys, model, polynomial_features)
    print(f"> Degree: {degree}, MSE: {mse}")


if __name__ == "__main__":
    # 観測データを作る。
    DATA_SIZE = 20
    xs, ys, random_xs, random_ys = make_dataset(n=DATA_SIZE, scale=0.4)
    draw_figure(xs, ys, random_xs, random_ys)

    # AIC,BICを計算する。
    MAX_DEGREE = 10
    make_aic_and_bic(random_xs, random_ys, DATA_SIZE, MAX_DEGREE)

    degree = 4
    execute_polynomial_regression(degree, xs, ys, random_xs, random_ys)
