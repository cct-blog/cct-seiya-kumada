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


# Update the function to use the Pochhammer symbol
def calculate_pi_term_lambda(n, lambd=1):
    a = (2 * n + 1) ** 2 / (4 * (n + lambd)) - n
    term = 1 / factorial(n) * (1 / (n + lambd) - 4 / (2 * n + 1)) * pochhammer(a, n - 1)
    return term


def calculate_pi_approx_lambda(limit, lambd=1):
    approx = 4 + sum(calculate_pi_term_lambda(n, lambd) for n in range(1, limit + 1))
    return approx


def calculate_diff_from_pi(values):
    return [np.pi - value for value in values]
    # return [abs(np.pi - value) for value in values]


def plot_values(pi_values_0, pi_values_1, pi_values_2, pi_values_3, new_limits, file_name, pi_value=True):
    # Plotting the results for both λ=1 and λ=2 with actual π value for reference as a red solid line
    plt.figure(figsize=(10, 6))
    plt.plot(new_limits, pi_values_0, "o-", label="Approximated π (λ=0)")
    plt.plot(new_limits, pi_values_1, "o-", label="Approximated π (λ=1)")
    plt.plot(new_limits, pi_values_2, "o-", label="Approximated π (λ=2)")
    plt.plot(new_limits, pi_values_3, "o-", label="Approximated π (λ=3)")
    if pi_value:
        plt.axhline(y=np.pi, color="r", linestyle="-", linewidth=2, label="True π")
    # plt.xscale("log")
    plt.ylim(3.135, 3.15)
    plt.xlabel("Number of terms in the series")
    plt.ylabel("Value of π")
    plt.title("Approximation of π using the given series with λ=1 and λ=2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/kumada/projects/cct-seiya-kumada/new_pi_formulation/images/{file_name}.jpg")


def plot_diffs(pi_values_0, pi_values_1, pi_values_2, pi_values_3, new_limits, file_name, pi_value=True):
    # Plotting the results for both λ=1 and λ=2 with actual π value for reference as a red solid line
    plt.figure(figsize=(10, 6))
    plt.plot(new_limits, pi_values_0, "o-", label="Approximated π (λ=0)")
    plt.plot(new_limits, pi_values_1, "o-", label="Approximated π (λ=1)")
    plt.plot(new_limits, pi_values_2, "o-", label="Approximated π (λ=2)")
    plt.plot(new_limits, pi_values_3, "o-", label="Approximated π (λ=3)")
    # plt.xscale("log")
    plt.ylim(-0.0021, 0.0021)
    plt.xlabel("Number of terms in the series")
    plt.ylabel("Value of π")
    plt.title("Approximation of π using the given series with λ=1 and λ=2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/kumada/projects/cct-seiya-kumada/new_pi_formulation/images/{file_name}.jpg")


if __name__ == "__main__":
    new_limits = list(range(1, 15))

    pi_values_0 = [calculate_pi_approx_lambda(limit, 0) for limit in new_limits]
    diff_0 = calculate_diff_from_pi(pi_values_0)
    pi_values_1 = [calculate_pi_approx_lambda(limit, 1) for limit in new_limits]
    diff_1 = calculate_diff_from_pi(pi_values_1)
    pi_values_2 = [calculate_pi_approx_lambda(limit, 2) for limit in new_limits]
    diff_2 = calculate_diff_from_pi(pi_values_2)
    pi_values_3 = [calculate_pi_approx_lambda(limit, 3) for limit in new_limits]
    diff_3 = calculate_diff_from_pi(pi_values_3)

    # file_name = "pi_values"
    # plot_values(pi_values_0, pi_values_1, pi_values_2, pi_values_3, new_limits, file_name)

    file_name = "diff_values"
    plot_diffs(diff_0, diff_1, diff_2, diff_3, new_limits, file_name, pi_value=False)
