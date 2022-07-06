#!/usr/bin/env python
# -*- coding:utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


def model(params, x):
    return params["a"] * x + params["b"]


def loss(params, x, y):
    y_pred = model(params, x)
    return jnp.power(y_pred - y, 2).mean()


# update "params"
def update(params, grads, lr):
    return jax.tree_map(lambda p, g: p - lr * g, params, grads[0])


def create_dataset(a, b, n, seed):
    np.random.seed(seed)
    x = np.random.rand(N)
    y = a * x + b + np.random.randn(n)
    return jnp.array(x), jnp.array(y)


# differentiate "loss" with its first argument
grad_loss = jax.grad(loss, argnums=[0])


@jax.jit
def train_(x, y, params):
    # d(loss)/dx
    grads = grad_loss(params, x, y)
    # update "params"
    params = update(params, grads, lr)
    return params


@jax.jit
def train(epochs, x, y, lr, params):
    def body_fun(idx, params):
        # d(loss)/dx
        grads = grad_loss(params, x, y)
        # update params
        params = update(params, grads, lr)
        return params

    params = jax.lax.fori_loop(0, epochs, body_fun, params)
    return params
    # "fori_loop" is equivalent to the following codes:
    # val = params
    # for i in range(0, epochs):
    #    val = body_fun(i, params)
    # return val


def display_line(x, y, params):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    xs = np.linspace(0, 1, 100)
    ys = model(params, xs)
    plt.plot(xs, ys)
    plt.savefig("./fig_with_jax.jpg")


if __name__ == "__main__":

    args = sys.argv

    # create dataset
    N = 100
    seed = 1
    true_a = 3
    true_b = 1
    x, y = create_dataset(true_a, true_b, N, seed)

    # initialize params
    params = {"a": jnp.array(1.0), "b": jnp.array(0.0)}

    # set hyperparameters
    epochs = 10000
    lr = 1.0e-3

    # train
    if len(args) == 2 and args[1] == "fori":
        # run using fori!
        start = time.time()
        params = train(epochs, x, y, lr, params)
        end = time.time()
    else:
        start = time.time()
        for _ in range(epochs):
            params = train_(x, y, params)
        end = time.time()

    print(f"{end - start}[sec]")

    # display results
    for (k, v) in params.items():
        print(k, v)

    display_line(x, y, params)
