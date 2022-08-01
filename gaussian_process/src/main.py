import pandas as pd
import scipy.stats as stats
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
import pymc3 as pm


import time

import matplotlib.pyplot as plt
import arviz as az

DATA_PATH = "./data/data.txt"
TRACE_PATH = "./trace.nc"
X_DIM = 10
TRAIN_SIZE = 300
SAMPLE_SIZE = 1000


def load_dataset(
    path: str, train_size: int
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    df = pd.read_table(path)
    ds = df.to_numpy()
    ys = ds[:, X_DIM]
    xs = ds[:, :X_DIM]
    xs = stats.zscore(xs)

    train_xs = xs[:TRAIN_SIZE, :]
    test_xs = xs[TRAIN_SIZE:, :]
    train_ys = ys[:TRAIN_SIZE]
    test_ys = ys[TRAIN_SIZE:]

    return train_xs, train_ys, test_xs, test_ys


def define_model(xs: NDArray[np.float32], ys: NDArray[np.float32]) -> Any:
    with pm.Model() as model:
        el = pm.Gamma("el", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)
        cov = eta**2 * pm.gp.cov.Matern52(X_DIM, el)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=xs)
        sigma = pm.HalfCauchy("sigma", beta=5)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=ys)
    return model, gp


if __name__ == "__main__":
    train_xs, train_ys, test_xs, test_ys = load_dataset(DATA_PATH, TRAIN_SIZE)
    print(f"train x shape: {train_xs.shape}, train y shape: {train_ys.shape}")
    print(f"test x shape: {test_xs.shape}, test y shape: {test_ys.shape}")
    model, gp = define_model(train_xs, train_ys)

    start = time.time()
    with model:
        trace = pm.sample(SAMPLE_SIZE, cores=1, return_inferencedata=True)
    end = time.time()
    print(f"{end - start}[sec]")
    # 322[sec]
    trace.to_netcdf(TRACE_PATH)
    trace = az.from_netcdf(TRACE_PATH)  # type:ignore
    with model:
        f_pred = gp.conditional("f_pred", test_xs)
        pred_samples = pm.sample_posterior_predictive(trace, var_names=["f_pred"], samples=100)

    pred_ys = pred_samples["f_pred"]
    pred_mean_ys = np.mean(pred_ys, axis=0)
    pred_std_ys = np.std(pred_ys, axis=0)
    plt.errorbar(test_ys, pred_mean_ys, yerr=pred_std_ys, fmt="o")
    plt.show()
