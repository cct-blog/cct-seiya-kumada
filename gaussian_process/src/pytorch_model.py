import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
from numpy.typing import NDArray
import random

H1, H2 = 10, 10
N_EPOCH = 5000


def torch_fix_seed(seed: int = 42) -> None:
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True  # type:ignore


class NN_Model(nn.Module):
    # 各層のコンポーネントを定義
    def __init__(self, input_dim: int, h1: int = H1, h2: int = H2):
        super(NN_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = F.relu

    # 生成過程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y: torch.Tensor = self.fc3(x)
        return y


def train(
    optimizer: Any,
    model_torch: Any,
    loss_func: Any,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
) -> List[float]:
    # 最適化
    loss_list = []
    for e in range(N_EPOCH):
        # 今のパラメータによるモデルで予測値を算出
        pred = model_torch(x_data)
        # 損失関数を再計算
        loss = loss_func(pred, y_data)
        # 勾配を0に初期化
        optimizer.zero_grad()
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()

        loss_list.append(loss.detach().numpy())
        if e % 100 == 0 and e != 0:
            print(f"{e}:{loss.item()}")
    return loss_list


def save_loss_graph(loss_list: List[float], loss_path: str) -> None:
    plt.plot(np.array(loss_list))
    plt.xlabel("step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.clf()


def predict(model_torch: Any, x_new: torch.Tensor) -> Any:
    return model_torch(x_new)


def save_predictions(
    pred_train_ys: torch.Tensor,
    gt_train_ys: torch.Tensor,
    pred_test_ys: torch.Tensor,
    gt_test_ys: torch.Tensor,
    path: str,
) -> None:
    pred_train_ys = pred_train_ys.detach().numpy()
    gt_train_ys = gt_train_ys.detach().numpy()
    pred_test_ys = pred_test_ys.detach().numpy()
    gt_test_ys = gt_test_ys.detach().numpy()
    _, ax = plt.subplots(figsize=(10, 5))
    # 訓練データ可視化
    ax.scatter(pred_train_ys, gt_train_ys, color="r", label="train")
    # テストデータ可視化
    ax.scatter(pred_test_ys, gt_test_ys, color="b", label="test")

    xvalues = np.linspace(0, 400, 100)
    yvalues = np.linspace(0, 400, 100)
    plt.plot(xvalues, yvalues, linestyle="dashed")

    ax.set_xlabel("$prediction$")
    ax.set_ylabel("$ground truth$")
    ax.legend()
    plt.savefig(path)


def execute(
    train_xs: NDArray[np.float32],
    train_ys: NDArray[np.float32],
    test_xs: NDArray[np.float32],
    test_ys: NDArray[np.float32],
    loss_path: str,
    pred_path: str,
    model_path: str,
) -> None:
    torch_fix_seed()
    _, x_dim = train_xs.shape
    model_torch = NN_Model(x_dim)

    # optimizerの設定
    optimizer = torch.optim.Adam(model_torch.parameters(), lr=0.03)

    # 損失関数
    loss_func = nn.MSELoss()

    train_torch_xs = torch.from_numpy(train_xs)
    train_torch_ys = torch.from_numpy(train_ys).unsqueeze(-1)

    # 訓練
    loss_list = train(
        optimizer, model_torch, loss_func, train_torch_xs, train_torch_ys
    )

    # 損失関数の可視化
    save_loss_graph(loss_list, loss_path)
    torch.save(model_torch.state_dict(), model_path)

    model_torch.load_state_dict(torch.load(model_path))  # type:ignore

    # 予測
    pred_train_ys = predict(model_torch, train_torch_xs)
    test_torch_xs = torch.from_numpy(test_xs)
    pred_test_ys = predict(model_torch, test_torch_xs)

    test_torch_ys = torch.from_numpy(test_ys).unsqueeze(-1)
    save_predictions(
        pred_train_ys, train_torch_ys, pred_test_ys, test_torch_ys, pred_path
    )
