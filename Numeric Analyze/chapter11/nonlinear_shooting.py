from collections.abc import Sequence
from numbers import Real
from types import FunctionType
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from chapter2.newton import secant
from chapter5.equation_system import ralston4_linear


def subroutine(
    f: Sequence[FunctionType],
    a: Real,
    b: Real,
    alpha: Real,
    N: int,
    t: Real = 0.0,
    method: FunctionType = ralston4_linear,
):
    """
    the function is a subroutine used to get the y(b,t)
    when t specified
    """
    h: Real = (b - a) / N
    y: List[Real] = [alpha, t]
    for i in range(N):
        y: List[Real] = method([*f], y, i * h, h)
    return y[0]


def nonlinear_shooting_secant(
    f: Sequence[FunctionType],
    a: Real,
    b: Real,
    alpha: Real,
    beta: Real,
    N: int,
    t0: Real = 0.0,
    t1: Real = 1.0,
    method: FunctionType = ralston4_linear,
    tol: Real = 1e-5,
    maxIter: int = 30,
    verbose: bool = False,
):
    """
    the nonlinear_shooting_secant method is use secant method
    to find the root of the function y(b,t) - beta = 0
    Compared with newton method, we don't need to solve another
    ode function to get y'(b,t), although the convergent rate
    might be slower

    :param f: linear equations after transforming
    :param t0: first slope guessed at starting point a
    :param t1: second slope guessed at starting point a
    """
    assert N > 1
    assert t0 != t1
    a, b = min(a, b), max(a, b)
    h: Real = (b - a) / N
    # to setup secant process, we need at least two point t0 and t1
    # for computational convenience, we choose 0 and 1 as t0 and t1
    # as by default where t means the slope at starting point a
    g_t = lambda t_: subroutine(f, a, b, alpha, N, t_, method) - beta
    # secant process
    t: Real = secant(g_t, t0, t1, tol, maxIter, verbose=verbose)
    # calculate y with t value
    y: List[Real] = [[alpha, t]]
    for i in range(N):
        y.append(method([*f], y[-1], h * i, h))
    # drop derivative term
    y: ndarray = np.asarray(y).T[0]
    return y


def nonlinear_shooting_newton(
    f: Sequence[FunctionType],
    f_y: FunctionType,
    f_y1: FunctionType,
    a: Real,
    b: Real,
    alpha: Real,
    beta: Real,
    N: int,
    t0: Real = 0.0,
    method: FunctionType = ralston4_linear,
    tol: Real = 1e-5,
    maxIter: int = 30,
    verbose: bool = False,
):
    """
    different from nonlinear_shooting_secant method, nonlinear_shooting_newton
    method has to know derivative between y(b,t) and t, to get this, we need
    to solve the ODE function z'' = f_y*z + f_y' * z', z'(a) = 0 and z'(a) = 1
    where z is the derivative between y(b,t) and t, thus we need to know f_y
    and f_y'(f_y1) which needed to input as parameters
    """
    assert N > 1
    a, b = min(a, b), max(a, b)
    h: Real = (b - a) / N
    return


def main():
    # f1 = lambda x_, y0_, y1_: y1_
    # f2 = lambda x_, y0_, y1_: 1 / 8 * (32 + 2 * x_ ** 3 - y0_ * y1_)
    # f_exact = lambda x_: x_ ** 2 + 16 / x_
    # a, b = 1.0, 2.0
    # alpha: Real = 17.0
    # beta: Real = 12.
    f1 = lambda x_, y0_, y1_: y1_
    f2 = lambda x_, y0_, y1_: 0.5 * (1 - y1_**2 - y0_ * np.sin(x_))
    f_exact = lambda x_: 2 + np.sin(x_)
    a, b = 0.0, np.pi
    alpha: Real = 2.0
    beta: Real = 2.0
    N: int = 30
    x = np.linspace(a, b, N + 1)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(10, 8), sharex="col"
    )

    y_exact = f_exact(x)
    axes[0].plot(x, y_exact, "--", label="exact value")

    y = nonlinear_shooting_secant(
        [f1, f2], a, b, alpha, beta, N, 1.0, 2.0, verbose=True, tol=1e-5
    )
    axes[0].plot(x, y, "o", ms=4, label="secant")

    err = np.abs(y - y_exact)
    axes[1].plot(x, err, "o--", ms=4, label="secant")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
