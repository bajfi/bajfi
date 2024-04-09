from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy import sparse

"""
    hyperbolic function has formation like
    d^2(u)/d(t)^2 = alpha * (d^2(u) / d(x)^2)
    where alpha >= 0, with boundary condition
    u(0,t) = 0, u(l,t) = 0
    u(x,0) = f(x), u'(x,0) = g(x)
"""


def wave(
    bc: Sequence[FunctionType, FunctionType],
    alpha: Real,
    x0: Real,
    x1: Real,
    t: Real,
    m: int,
    n: int,
):
    """
    :param bc: boundary value and first order derivative
    :param alpha: the rhs coefficient, non-negative
    :param m: sub-intervals in t-axis
    :param n: sub-intervals in x-axis
    :param t: maximum time
    """
    assert alpha >= 0
    assert m > 2
    assert n > 2
    x0, x1 = min(x0, x1), max(x0, x1)
    h: Real = (x1 - x0) / n
    k: Real = t / m
    xx: ndarray = np.linspace(x0, x1, n + 1)[1:-1]
    lbd_sq: Real = alpha * k**2 / h**2
    # calculate the value of first time step
    data: ndarray = np.zeros((m + 1, n + 1), dtype=np.float_)  # store result
    w0: ndarray = bc[0](xx)
    data[0, 1:-1] = w0
    mat: ndarray = sparse.diags(
        (lbd_sq / 2, 1 - lbd_sq, lbd_sq / 2), (-1, 0, 1), shape=(n - 1, n - 1)
    ).toarray()
    w1: ndarray = mat @ w0 + k * bc[1](xx)
    data[1, 1:-1] = w1
    # propagation
    a: ndarray = sparse.diags(
        (lbd_sq, 2 * (1 - lbd_sq), lbd_sq), (-1, 0, 1), shape=(n - 1, n - 1)
    ).toarray()
    for i in range(2, m + 1):
        data[i, 1:-1] = a @ data[i - 1, 1:-1] - data[i - 2, 1:-1]
    return data


def main():
    fx = lambda x_: np.sin(np.pi * x_)
    gx = lambda x_: np.zeros_like(x_, dtype=np.float_)
    alpha = 4.0
    x0, x1 = 0.0, 1.0
    n = 10
    m = 20
    t = 1.0
    xx = np.linspace(x0, x1, n + 1)
    k = t / m

    data = wave([fx, gx], alpha, x0, x1, t, m, n)
    for i in range(m + 1):
        plt.plot(xx, data[i], "o--", lw=1, ms=2, label=f"{round(i * k, 2)}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
