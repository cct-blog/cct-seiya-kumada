import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


def load_data(path: str, publisher_name: str) -> pd.DataFrame:
    # read only columns = ['Publisher', 'Year', 'Global_Sales']
    df = pd.read_csv(path, usecols=["Publisher", "Year", "Global_Sales"])
    df = df.dropna(how="any")

    df = df[df["Publisher"] == publisher_name]
    # sort by Year
    df = df.sort_values(by=["Year"])
    # replace column Platform and Year with index
    df = df.set_index(["Publisher", "Year"])
    # unify JP_Sales values for the same Year
    df = df.groupby(level=["Publisher", "Year"]).sum()
    return df


def draw_plot(df: pd.DataFrame, title: str, path: str) -> None:
    # plot Year vs JP_Sales
    plt.figure(figsize=(24, 4))
    plt.plot(df.index.get_level_values("Year"), df["Global_Sales"])
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Global_Sales")
    plt.savefig(path)
    plt.clf()


def calculate_kpss(ts: pd.DataFrame) -> np.float64:
    # KPSS検定
    _, p_value, _, _ = sm.tsa.kpss(ts, nlags=1)
    return p_value  # type:ignore


def calculate_diff(ts: pd.DataFrame, periods: int) -> pd.DataFrame:
    diff_1 = ts.diff(periods=periods)
    # １地点目がNaN(0地点はないため)になるため、dropna()
    diff_1 = diff_1.dropna()
    return diff_1


def evaluate_kpss(df: pd.DataFrame) -> tuple[np.float64, np.float64]:
    kpss_0 = calculate_kpss(df["Global_Sales"])
    df_diff_1 = calculate_diff(df, 1)
    kpss_1 = calculate_kpss(df_diff_1["Global_Sales"])
    return kpss_0, kpss_1


def calculate_auto_correlation(lags: int, ticks: int, ts: pd.Series, path: str, title: str) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    # add ticks to the plot
    ax1.set_xticks(range(0, lags + 1, ticks))
    # add title to the plot
    ax1.set_xlabel(title)
    fig = sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    plt.savefig(path)


Tuple4Int = tuple[int, int, int, int]
Tuple3Int = tuple[int, int, int]


def execute_grid_search(
    ts_train: pd.DataFrame, d: int, m: int
) -> list[Tuple3Int, Tuple4Int, np.float64]:
    # SARIMA(p,d,q)(sp,sd,sq)[s]の次数の範囲を決める。
    # 範囲は計算量を減らすため、経験上、p,d,qを0～2、sp,sd,sqを0～1くらいに限定する。
    p = q = range(0, 3)
    if m == 0:
        sp = sd = sq = range(0, 1)
    else:
        sp = sd = sq = range(0, 2)

    # グリッドサーチのために、p,q,sp,sd,sqの組み合わせのリストを作成する。
    # 定常性の確認よりd=1,周期sは決め打ちで7としている。
    pdq = [(x[0], d, x[1]) for x in list(itertools.product(p, q))]
    seasonal_pdq = [(x[0], x[1], x[2], m) for x in list(itertools.product(sp, sd, sq))]

    best_result = [0, 0, 10000000]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(
                    ts_train,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                results = mod.fit()
                print("order{}, s_order{} - AIC: {}".format(param, param_seasonal, results.aic))

                if results.aic < best_result[2]:
                    best_result = [
                        param,  # type:ignore
                        param_seasonal,  # type:ignore
                        results.aic,
                    ]
            except Exception:
                print("Exception!")
                continue
    return best_result


def read_params(
    path: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with open(path, "r") as f:
        lines = f.readlines()
        items = tuple([int(x) for x in lines[0].split(",")])
        a = items[:3]
        A = items[3:]

    return a, A


def train(ts_train: pd.DataFrame, a: tuple[int, ...], A: tuple[int, ...]) -> SARIMAXResultsWrapper:
    (p, d, q) = a
    (P, D, Q, m) = A
    model = SARIMAX(endog=ts_train, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit()
    return model


def predict(
    arima_model: SARIMAXResultsWrapper, df_test: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    # predict
    train_pred = arima_model.predict()
    test_pred = arima_model.forecast(len(df_test))
    test_pred_ci = arima_model.get_forecast(len(df_test)).conf_int()
    return (train_pred, test_pred, test_pred_ci)


def draw_results(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_pred: pd.Series,
    test_pred: pd.Series,
    test_pred_ci: pd.DataFrame,
    path: str,
) -> None:
    # グラフ化
    plt.figure(figsize=(24, 4))
    # fig, ax = plt.subplots()
    plt.plot(
        df_train.index.get_level_values("Year"),
        df_train["Global_Sales"],
        label="observed(train dataset)",
        color="red",
        marker="o",
    )

    plt.plot(
        df_test.index.get_level_values("Year"),
        df_test["Global_Sales"],
        label="observed(test dataset)",
        color="green",
        marker="o",
    )
    plt.plot(df_train.index.get_level_values("Year"), train_pred.values, color="blue", marker="o")
    plt.plot(
        df_test.index.get_level_values("Year"),
        test_pred.values,
        label="SARIMA",
        # alpha=0.5,
        color="blue",
        marker="o",
    )
    plt.fill_between(
        df_test.index.get_level_values("Year"),
        test_pred_ci.iloc[:, 0],
        test_pred_ci.iloc[:, 1],
        color="gray",
        alpha=0.2,
    )
    plt.ylabel("Number of Orders")  # タテ軸のラベル
    plt.xlabel("Day")  # ヨコ軸のラベル
    # y=0の線を引く
    plt.axhline(y=0, color="black", linestyle="-")
    plt.legend()
    plt.savefig(path)
    plt.clf()


def draw_results_part(
    df_test: pd.DataFrame, test_pred: pd.Series, test_pred_ci: pd.DataFrame, path: str
) -> None:
    # グラフ化
    plt.figure(figsize=(24, 4))
    # fig, ax = plt.subplots()

    plt.plot(
        df_test.index.get_level_values("Year"),
        df_test["Global_Sales"],
        label="observed(test dataset)",
        color="green",
        marker="o",
    )
    plt.plot(
        df_test.index.get_level_values("Year"),
        test_pred.values,
        label="SARIMA",
        alpha=0.5,
        color="blue",
        marker="o",
    )
    plt.fill_between(
        df_test.index.get_level_values("Year"),
        test_pred_ci.iloc[:, 0],
        test_pred_ci.iloc[:, 1],
        color="gray",
        alpha=0.2,
    )
    plt.ylabel("Number of Orders")  # タテ軸のラベル
    plt.xlabel("Day")  # ヨコ軸のラベル
    # y=0の線を引く
    plt.axhline(y=0, color="black", linestyle="-")
    plt.legend()
    plt.savefig(path)
    plt.clf()
