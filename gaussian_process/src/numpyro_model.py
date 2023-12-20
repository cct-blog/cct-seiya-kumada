import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist

# from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive

# 隠れ層の次元
H1, H2 = 10, 10


# relu関数
def relu(x):
    return jnp.maximum(x, 0.0)


def model(X, Y, h1, h2):
    # バイアス付与
    X = jnp.power(X, jnp.arange(2))  # [X^0,X^1]=[1,X]
    D_X = X.shape[1]  # X.shape = (N,D)

    # 第1層の重みをサンプリング
    w1 = numpyro.sample(
        "w1", dist.Normal(jnp.zeros((D_X, h1)), 10.0 * jnp.ones((D_X, h1))).to_event(2)
    )
    # w1.shape = (D,H1)
    # 第1層の線形結合と非線形変換
    z1 = relu(jnp.matmul(X, w1))  # (N,H1)

    # 第2層の重みをサンプリング
    w2 = numpyro.sample(
        "w2", dist.Normal(jnp.zeros((h1, h2)), 10.0 * jnp.ones((h1, h2))).to_event(2)
    )
    # w2.shape = (H1,H2)
    # 第2層の線形結合と非線形変換
    z2 = relu(jnp.matmul(z1, w2))  # (N,H2)

    # 出力層の重みをサンプリング
    w3 = numpyro.sample(
        "w3", dist.Normal(jnp.zeros((h2, 1)), 10.0 * jnp.ones((h2, 1))).to_event(2)
    )
    # w3.shape = (H2,1)
    # 出力層の線形結合と非線形変換
    z3 = jnp.matmul(z2, w3)  # (N,1)

    # 観測ノイズの標準偏差をサンプリング
    sigma_obs = numpyro.sample("noise_obs", dist.Uniform(0.0, 2.0))
    # 尤度の積
    with numpyro.plate("data", X.shape[0]):
        _ = numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)

    # 全ての軸をevent_shapeにしても計算はできる
    # obs = numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(2), obs=Y)


# NUTによるMCMCの設定
def run_inference(model, rng_key, X, Y, h1, h2):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
    )
    mcmc.run(rng_key, X, Y, h1, h2)
    return mcmc.get_samples()


def save_predictions(
    predictive,
    rng_key_predict,
    x_data,
    y_data,
    x_linspace,
    y_linspace,
    x_new,
    model_path,
):

    # 新規データ
    x_new = jnp.linspace(-2.0, 2.0, 1000)[:, jnp.newaxis]
    # 新規データを入力して予測分布を出力
    y_pred_samples = predictive(rng_key_predict, X=x_new, Y=None, h1=H1, h2=H2)["Y"]
    # 予測分布のからのサンプルの平均
    y_pred_mean = y_pred_samples.mean(axis=0)
    # 予測分布からのサンプルの90パーセンタイル
    percentiles = np.percentile(y_pred_samples.squeeze(-1), [5.0, 95.0], axis=0)

    _, ax = plt.subplots(figsize=(10, 5))
    # データ可視化
    ax.plot(x_data, y_data, "o", markersize=3, label="data")
    # 真の関数
    ax.plot(x_linspace, y_linspace, label="true_func")
    # 予測分布の平均
    ax.plot(x_new, y_pred_mean, label="mean")
    # 予測分布の90パーセンタイル
    ax.fill_between(
        x_new.squeeze(-1),
        percentiles[0, :],
        percentiles[1, :],
        alpha=0.5,
        label="90percentile",
        color="orange",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(-13, 13)
    ax.legend()
    plt.savefig(model_path)


def execute(x_data, y_data, x_linspace, y_linspace, x_new, loss_path, model_path):
    # データをjax.numpy型に変換
    x_data = jnp.array(x_data)[:, jnp.newaxis]
    y_data = jnp.array(y_data)[:, jnp.newaxis]

    # 疑似乱数生成器
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    # 推論の実行（y_dataのshapeを(30, )から(30, 1)にする））
    samples = run_inference(model, rng_key, x_data, y_data, H1, H2)

    # MCMCサンプルを利用した予測分布
    predictive = Predictive(model, samples)

    save_predictions(
        predictive,
        rng_key_predict,
        x_data,
        y_data,
        x_linspace,
        y_linspace,
        x_new,
        model_path,
    )
