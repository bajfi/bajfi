from math import factorial
from numbers import Real
from types import FunctionType
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def high_order(derivative: List[FunctionType], init: Real, t_pre: Real, h: Real):
    """
    derivative: derivative function of each order,begin with order 1
    a,b: calculation interval
    N: number of steps
    order:the order to be evaluated
    init: initial value of the function
    """
    assert h > 0
    f = lambda t, y: sum(
        h**i * derivative[i - 1](t, y) / factorial(i)
        for i in range(1, len(derivative) + 1)
    )
    yy = init + f(t_pre, init)
    return yy


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1
    f_1 = lambda t_, y_: y_ - t_**2 - 2 * t_ + 1
    f_2 = lambda t_, y_: y_ - t_**2 - 2 * t_ - 1
    f_3 = lambda t_, y_: y_ - t_**2 - 2 * t_ - 1
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0, 4
    N = 20
    h = (b - a) / N
    init = 0.5
    t = np.linspace(a, b, N + 1)
    yy = [init]
    for i in range(N):
        yy.append(high_order([f_0, f_1, f_2, f_3], yy[-1], t[i], h))

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="row"
    )
    axes[0].plot(t, yy, "ro", ms=2, label="high order method")
    axes[0].plot(t, f_exact(t), ":", label="exact value")
    axes[0].legend()

    err = np.abs(f_exact(t) - yy)
    axes[1].plot(t, err, "o--", ms=2, label="error")

    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
