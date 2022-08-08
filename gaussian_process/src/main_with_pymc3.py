# import pandas as pd
# import scipy.stats as stats
from typing import Any
import numpy as np
from numpy.typing import NDArray
import pymc3 as pm
import src.util as util

import time

import matplotlib.pyplot as plt
import arviz as az

TRACE_PATH = "./trace.nc"
SAMPLE_SIZE = 1000
OUTPUT_PATH = "./result_with_pymc3.jpg"


def define_model(xs: NDArray[np.float32], ys: NDArray[np.float32]) -> Any:
    _, x_dim = xs.shape
    with pm.Model() as model:
        el = pm.Gamma("el", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)
        cov = eta**2 * pm.gp.cov.Matern52(x_dim, el)
        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=xs)
        sigma = pm.HalfCauchy("sigma", beta=5)
        # cov = sigma**2 * np.eye(util.TRAIN_SIZE, util.TRAIN_SIZE)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=ys)
        # pm.MvNormal("y", mu=f, cov=cov, observed=ys)
    return model, gp


def train(model: Any, gp: Any) -> Any:
    start = time.time()
    with model:
        trace = pm.sample(SAMPLE_SIZE, cores=1, return_inferencedata=True)
    end = time.time()
    print(f"{end - start}[sec]")
    trace.to_netcdf(TRACE_PATH)
    return trace


def predict(train_xs: Any) -> Any:
    trace = az.from_netcdf(TRACE_PATH)  # type:ignore
    with model:
        gp.conditional("f_pred", train_xs)
        pred_samples = pm.sample_posterior_predictive(
            trace, var_names=["f_pred"], samples=100
        )
    return pred_samples


def save_fig(pred_samples: Any, ys: Any, output_path: str) -> None:
    pred_ys = pred_samples["f_pred"]
    pred_mean_ys = np.mean(pred_ys, axis=0)
    pred_std_ys = np.std(pred_ys, axis=0)
    plt.errorbar(ys, pred_mean_ys, yerr=pred_std_ys, fmt="o")
    xvalues = np.linspace(0, 400, 100)
    yvalues = np.linspace(0, 400, 100)
    plt.plot(xvalues, yvalues, linestyle="dashed")
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.legend(loc="best")
    plt.savefig(output_path)


if __name__ == "__main__":
    # load dataset
    (
        train_xs,
        train_ys,
        test_xs,
        test_ys,
    ) = util.load_dataset_with_high_correlation(util.DATA_PATH, util.TRAIN_SIZE)

    # define model
    model, gp = define_model(train_xs, train_ys)

    # train
    trace = train(model, gp)

    # predict
    pred_samples = predict(train_xs)

    # save result
    save_fig(pred_samples, train_ys, OUTPUT_PATH)
