#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


def model(x, a, b):
    return a * x + b


def loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()


def zero_grad(a, b):
    a.grad = torch.zeros(1)
    b.grad = torch.zeros(1)


def update(a, b, lr=1e-3):
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    return a, b


def create_dataset(a, b, n, seed):
    np.random.seed(seed)
    x = np.random.rand(N)
    y = a * x + b + 0.5 * np.random.randn(n)
    return torch.tensor(x), torch.tensor(y)


def display_x_and_y(x, y):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("./fig_1.jpg")


def display_line(x, y, a, b):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    xs = np.linspace(0, 1, 100)
    ys = model(xs, a, b)
    plt.plot(xs, ys)
    plt.savefig("./fig_2.jpg")



if __name__ == "__main__":
    # create dataset
    N = 100
    seed = 1
    true_a = 3
    true_b = 1
    x, y = create_dataset(true_a, true_b, N, seed)

    # display_x_and_y(x, y)

    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    epochs = 10000
    start = time.time()
    for i in range(epochs):
        zero_grad(a, b)
        y_pred = model(x, a, b)
        loss_value = loss(y_pred, y)
        loss_value.backward()
        update(a, b, lr=1e-2)
    end = time.time()
    print(f"{end - start}[sec]")

    print(f"a {a.item()}")
    print(f"b {b.item()}")

    display_line(x, y, a.detach().numpy(), b.detach().numpy())
