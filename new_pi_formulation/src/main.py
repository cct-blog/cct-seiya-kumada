from math import factorial

import matplotlib.pyplot as plt
import numpy as np

# https://x.com/wakapaijazz/status/1804378231631437953


# Define the Pochhammer symbol
def pochhammer(x, n):
    result = 1
    for j in range(n):
        result *= x + j
    return result


def calculate_pi_term_lambda(n, lambd=1):
    a = (2 * n + 1) ** 2 / (4 * (n + lambd)) - n
    term = 1 / factorial(n) * (1 / (n + lambd) - 4 / (2 * n + 1)) * pochhammer(a, n - 1)
    return term


def calculate_pi_approx_lambda(limit, lambd=1):
    approx = 4 + sum(calculate_pi_term_lambda(n, lambd) for n in range(1, limit + 1))
    return approx


def calculate_diff_from_pi(values):
    return [np.abs(np.pi - value) for value in values]


def plot_values(ys_list, xs, file_name, pi_value=True):
    plt.figure(figsize=(10, 6))
    for lbd, ys in ys_list:
        plt.plot(xs, ys, "o-", label=f"Approximated π (λ={lbd})")

    # 軸の目盛りのフォントを大きくしたい。
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.hlines([np.pi], xmin=1, xmax=15, linestyles="dashed")
    plt.legend()
    plt.savefig(f"/home/kumada/projects/cct-seiya-kumada/new_pi_formulation/images/{file_name}.jpg")


def plot_diffs(ys_list, xs, file_name, pi_value=True):
    plt.figure(figsize=(10, 6))
    for lbd, ys in ys_list:
        plt.plot(xs, ys, "o-", label=f"Approximated π (λ={lbd})")

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.hlines([0], xmin=1, xmax=15, linestyles="dashed")
    plt.legend()
    plt.savefig(f"/home/kumada/projects/cct-seiya-kumada/new_pi_formulation/images/{file_name}.jpg")


def save_diff(diff, path):
    with open(path, "w") as f:
        for v in diff:
            f.write(f"{v}\n")


if __name__ == "__main__":
    xs = list(range(1, 15))

    ys_list = []
    diff_list = []
    for lbd in [0, 1, 2, 3]:
        ys = [calculate_pi_approx_lambda(x, lbd) for x in xs]
        diff = calculate_diff_from_pi(ys)
        ys_list.append((lbd, ys))
        diff_list.append((lbd, diff))

    file_name = "pi_values"
    plot_values(ys_list, xs, file_name)

    file_name = "diff_values"
    plot_diffs(diff_list, xs, file_name, pi_value=False)

    diffs = []
    for _, diff in diff_list:
        diffs.append(diff)
    diffs = np.array(diffs)
    path = "/home/kumada/projects/cct-seiya-kumada/new_pi_formulation/images/diff.csv"
    np.savetxt(path, diffs.T, delimiter=",")
