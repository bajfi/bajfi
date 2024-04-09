from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np


def euler(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    p267
    :param: f: rhs function
    :param: a,b: calculation interval
    :param: N: number of steps
    :param: init: initial value of the function
    """
    assert h > 0
    yy = init + h * f(t_pre, init)
    return yy


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    f_exact_2 = lambda t_: 2 - 0.5 * np.exp(t_)
    a, b = 0, 4
    N = 20
    h = (b - a) / N
    init = 0.5
    t = np.linspace(a, b, N + 1)
    yy = [init]
    for i in range(N):
        yy.append(euler(f_0, yy[-1], t[i], h))

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    axes[0].plot(t, yy, "ro", ms=1.5, label="Euler method")
    axes[0].plot(t, f_exact(t), label="exact value")

    err = np.abs(f_exact(t) - yy)
    axes[1].plot(t, err, "ro--", ms=2, label="error")
    axes[1].axline((0, 0), slope=1, label="linear growth", ls="--")

    M = np.max(np.abs(f_exact_2(t)))
    L = 1  # L is depend on specific problem
    f_err = lambda t: (b - a) / N * M / 2 / L * (np.exp(L * (t - a)) - 1)
    axes[1].plot(t, f_err(t), "--", label="error bound")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
