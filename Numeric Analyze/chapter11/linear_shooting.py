from numbers import Real
from types import FunctionType
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from chapter5.equation_system import ralston4_linear, rk4_linear


def linear_shoot(
    px: FunctionType,
    qx: FunctionType,
    rx: FunctionType,
    a: Real,
    b: Real,
    alpha: Real,
    beta: Real,
    N: int,
    method: FunctionType = ralston4_linear,
):
    """
    p674
    linear shooting method used to solve problems like
    y'' = p(x)*y' + q(x)*y + r(x)   y(a)=alpha, y(b) = beta
    by solving sub-problems
    1. y'' = p(x)*y' + q(x)*y + r(x)   y(a)=alpha, y'(a) = 0
    2. y'' = p(x)*y' + q(x)*y + r(x)   y(a)=0, y'(a) = 1
    which has solution y1(x) and y2(x) individually
    the final result of the problem is
    y(x) = y1(x) + (beta - y1(b))/y2(b) * y2(x)

    ralston_4 method is use as default here for it high precision,
    rk4_linear method is also available
    """
    assert N > 1
    a, b = min(a, b), max(a, b)
    h: Real = (b - a) / N
    # y1(x) construction
    y1_1 = lambda x_, y0_, y1_: y1_
    y1_2 = lambda x_, y0_, y1_: px(x_) * y1_1(x_, y0_, y1_) + qx(x_) * y0_ + rx(x_)
    y1: List[List[Real]] = [[alpha, 0.0]]
    # y2(x) construction
    y2_1 = lambda x_, y0_, y1_: y1_
    y2_2 = lambda x_, y0_, y1_: px(x_) * y2_1(x_, y0_, y1_) + qx(x_) * y0_
    y2: List[List[Real]] = [[0.0, 1.0]]
    # iteration process
    for i in range(N):
        y1.append(method([y1_1, y1_2], y1[-1], i * h, h))
        y2.append(method([y2_1, y2_2], y2[-1], i * h, h))
    # drop the derivative terms
    y1: ndarray = np.asarray(y1, dtype=np.float_).T[0]
    y2: ndarray = np.asarray(y2, dtype=np.float_).T[0]
    # calculate final result
    y: ndarray = y1 + (beta - y1[-1]) / y2[-1] * y2
    return y


def main():
    px = lambda x_: np.zeros_like(x_)
    qx = lambda x_: -4.0 * np.ones_like(x_)
    rx = lambda x_: np.cos(x_)
    a, b = 0.0, np.pi / 4
    alpha, beta = 0.0, 0.0
    N = 10
    x = np.linspace(a, b, N + 1)[1:-1]

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(10, 8), sharex="col"
    )

    y_exact = -1 / 3 * np.cos(2 * x) - 2**0.5 / 6 * np.sin(2 * x) + 1 / 3 * np.cos(x)
    axes[0].plot(x, y_exact, "--", label="exact value")

    for method in [rk4_linear, ralston4_linear]:
        y_shoot: ndarray = linear_shoot(
            px, qx, rx, a, b, alpha, beta, N, method=method
        )[1:-1]
        axes[0].plot(x, y_shoot, "o", ms=3, label=method.__name__)

        err_shoot = np.abs(y_shoot - y_exact)
        axes[1].plot(x, err_shoot, "o--", ms=3, label=method.__name__)

    axes[1].set_yscale("log")
    axes[0].legend()
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
