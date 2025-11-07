from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Use workspace-local folders for temporary files and artifacts.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
TRAIN_PLOT_PATH = PLOTS_DIR / "train_pred_vs_actual.png"
TEST_PLOT_PATH = PLOTS_DIR / "pred_vs_actual.png"
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 1000
MAX_DEPTH = 6
LEARNING_RATE = 0.05


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing regression dataset."""
    housing = fetch_california_housing(as_frame=True, data_home=str(DATA_DIR))
    return housing.data, housing.target  # type: ignore


def train_model(X_train, y_train) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor with defaults tuned for small tabular data."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        n_jobs=1,  # avoid sandbox issues with forked processes
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: xgb.XGBRegressor, X, y, split_name: str):
    predictions = model.predict(X)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)
    mape = np.mean(np.abs((y - predictions) / y)) * 100
    print(f"[{split_name}] RMSE: {rmse:.4f}")
    print(f"[{split_name}] R^2 : {r2:.4f}")
    print(f"[{split_name}] MAPE: {mape:.4f}%")
    return predictions, rmse, r2


def save_prediction_plot(y_true, y_pred, output_path: Path, title: str) -> None:
    """Scatter actual vs predicted values and persist the figure."""
    y_true_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    min_lim = float(min(y_true_arr.min(), y_pred_arr.min()))
    max_lim = float(max(y_true_arr.max(), y_pred_arr.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.7, s=30)
    ax.plot([min_lim, max_lim], [min_lim, max_lim], "r--", label="Ideal fit")
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved prediction plot → {output_path.relative_to(ROOT_DIR)}")


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost Regression on California Housing Dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="Path to the CSV file containing the dataset. If not provided, the built-in dataset will be used.",
    )
    return parser.parse_args()


def main() -> None:
    args = extract_args()
    if args.input_path:
        # CSVファイルからデータセットを読み込む。
        data = pd.read_csv(args.input_path)
        X = data.drop(columns=["MedHouseVal"])
        y = data["MedHouseVal"]
    else:
        # 組み込みのCalifornia Housingデータセットを読み込む。
        X, y = load_dataset()

    # 欠損値有無の確認。もしあれば警告を出す。
    if X.isnull().sum().any() or y.isnull().sum() > 0:
        print("警告: 欠損値が検出されました。")
    else:
        print("データセットに欠損値はありません。")

    # データの総数
    print(f"データセットの総数: {len(X)} サンプル")

    # 欠損値の割合
    total_missing = X.isnull().sum().sum() + y.isnull().sum()
    total_values = X.size + y.size
    missing_ratio = (total_missing / total_values) * 100
    print(f"欠損値の割合: {missing_ratio:.2f}%")

    # yの最小値と最大値
    print(f"ターゲット変数の最小値: {y.min():.4f}")
    print(f"ターゲット変数の最大値: {y.max():.4f}")

    # y>=5.0のサンプル数の割合を表示
    high_value_ratio = (y >= 5.0).mean() * 100
    print(f"ターゲット変数が5.0以上のサンプルの割合: {high_value_ratio:.2f}%")

    # y>=5.0のサンプルを削除する。
    mask = y < 5.0
    X = X[mask]
    y = y[mask]

    # データの80%を訓練用、20%をテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # モデルの訓練
    model = train_model(X_train, y_train)

    # モデルの評価
    train_pred, _, _ = evaluate(model, X_train, y_train, "train")
    test_pred, _, _ = evaluate(model, X_test, y_test, "test")

    # 予測結果のプロットを保存
    save_prediction_plot(y_train, train_pred, TRAIN_PLOT_PATH, "Train: Predicted vs Actual")
    save_prediction_plot(y_test, test_pred, TEST_PLOT_PATH, "Test: Predicted vs Actual")


if __name__ == "__main__":
    main()
