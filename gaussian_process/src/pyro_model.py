import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

# import torch.nn.functional as F
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

# from pyro.nn import DenseNN, PyroModule, PyroSample
from pyro.nn import PyroModule, PyroSample

H1, H2 = 10, 10

# event_shape:確率変数の次元
# batch_shape:パラメータが異なる複数の確率分布をまとめて扱う場合の確率分布の数


class Model(PyroModule):
    # 各コンポーネントを定義
    def __init__(self, h1=H1, h2=H2):
        super().__init__()
        # 第1層（確率変数のbatchshapeをeventshapeにする）
        self.fc1 = PyroModule[nn.Linear](1, h1)
        # 重みの事前分布
        # expand([h1,1])によりh1個の独立した1変数ガウス関数ができる。
        # これをh1次元の１つのガウス関数に変換する。
        self.fc1.weight = PyroSample(dist.Normal(0.0, 10.0).expand([h1, 1]).to_event(2))
        # h1個の1変数ガウス関数を1個のh1次元のガウス関数に変換する。
        self.fc1.bias = PyroSample(dist.Normal(0.0, 10.0).expand([h1]).to_event(1))

        # 第2層
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0.0, 10.0).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0.0, 10.0).expand([h2]).to_event(1))
        # 出力層
        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(0.0, 10.0).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0.0, 10.0).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    # データの生成過程を記述
    def forward(self, X, Y=None, h1=H1, h2=H2):
        # ニューラルネットワークの出力
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        # NNでガウス分布の平均を求める。
        mu = self.fc3(X)
        # 観測ノイズの標準偏差をサンプリング。これは学習しない。その都度サイコロをふる。
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 2.0))
        # 尤度の積
        with pyro.plate("data", X.shape[0]):  # 独立同分布であることを表現している。
            # shapeが(N, 1)であるため, 右の1をeventshapeにする
            obs = pyro.sample("Y", dist.Normal(mu, sigma).to_event(1), obs=Y)
        # 全ての軸をevent_shapeにしても計算はできる
        # obs = pyro.sample("Y", dist.Normal(mu, sigma).to_event(2), obs=Y)
        return mu


def save_loss_graph(loss_list, loss_path):
    plt.plot(np.array(loss_list))
    plt.xlabel("step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.clf()


def predict(model, guide):
    # 近似分布からのサンプルを利用した予測分布
    predictive = Predictive(model, guide=guide, num_samples=500)

    # 新規データ
    x_new = torch.linspace(-2.0, 2.0, 1000).unsqueeze(-1)
    # 新規データを入力して予測分布を出力
    y_pred_samples = predictive(x_new, None, H1, H2)["Y"]
    # 予測分布のからのサンプルの平均
    y_pred_mean = y_pred_samples.mean(axis=0)
    # 予測分布からのサンプルの90パーセンタイル
    percentiles = np.percentile(y_pred_samples.squeeze(-1), [5.0, 95.0], axis=0)
    return y_pred_mean, percentiles


def save_predictions(y_pred_mean, percentiles, x_data, y_data, x_linspace, y_linspace, x_new, path):
    _, ax = plt.subplots(figsize=(10, 5))
    # データ可視化
    ax.plot(x_data, y_data, "o", markersize=3, label="data")
    # 真の関数
    ax.plot(x_linspace, y_linspace, label="true_func")
    # 予測分布の平均
    ax.plot(x_new, y_pred_mean, label="mean")
    # 予測分布の90パーセンタイル
    ax.fill_between(x_new.squeeze(-1), percentiles[0, :], percentiles[1, :], alpha=0.5, label="90percentile", color="orange")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(-20, 20)
    ax.legend()
    plt.savefig(path)


def execute(xs, ys, x_linspace, y_linspace, new_xs, loss_path, model_path):
    model = Model()
    # パラメータをリセット
    pyro.clear_param_store()
    # 近似分布の設定
    guide = AutoDiagonalNormal(model)
    # optimizerの設定
    adam = pyro.optim.Adam({"lr": 0.03})
    # SVIクラスのインスタンス化
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    # データをtensorに変換
    x_data = torch.from_numpy(xs).float().unsqueeze(-1)
    y_data = torch.from_numpy(ys).float().unsqueeze(-1)

    # 最適化
    torch.manual_seed(0)
    n_epoch = 10000
    loss_list = []
    for epoch in range(n_epoch):

        # 変分推論の最適化ステップ
        loss = svi.step(x_data, y_data, H1, H2)
        loss_list.append(loss)

    # 損失関数の可視化
    save_loss_graph(loss_list, loss_path)

    y_pred_mean, percentiles = predict(model, guide)

    x_new = torch.from_numpy(new_xs).float().unsqueeze(-1)
    save_predictions(y_pred_mean, percentiles, x_data, y_data, x_linspace, y_linspace, x_new, model_path)
