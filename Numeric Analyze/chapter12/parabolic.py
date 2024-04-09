import warnings
from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy import sparse

from chapter6.factorization import crout

"""
    parabolic function has formation like
    du/dt = alpha * (d^2(u) / d(x)^2)
    where alpha >= 0
"""


def parabolic_forward(
    bc: Sequence[FunctionType, FunctionType],
    alpha: Real,
    x0: Real,
    x1: Real,
    init: Sequence[Real],
    t: Real,
    k: Real,
):
    """
    forward difference is fast but unstable, to get a stable solution,
    make sure alpha * t/h^2 <= 0.5

    :param bc: boundary condition
    :param alpha: the rhs coefficient, non-negative
    :param init: initial values
    param t: current time
    :param k: time step
    """
    assert alpha >= 0
    init: ndarray = np.asarray(init, dtype=np.float_)
    n: int = init.shape[0]
    assert n > 3
    assert k > 0
    x0, x1 = min(x0, x1), max(x0, x1)
    h: Real = (x1 - x0) / (n - 1)
    if alpha * k / h**2 > 0.5:
        warnings.warn("To get a stable solution, make sure alpha * t/h^2 <= 0.5")
    lbd: Real = alpha * k / h**2
    # setup iteration matrix
    a: ndarray = sparse.diags(
        (lbd, 1 - 2 * lbd, lbd), (-1, 0, 1), shape=(n - 2, n - 2)
    ).toarray()
    y: ndarray = a @ init[1:-1]
    return np.pad(y, (1, 1), constant_values=(bc[0](t), bc[1](t)))


def parabolic_backward(
    bc: Sequence[FunctionType, FunctionType],
    alpha: Real,
    x0: Real,
    x1: Real,
    init: Sequence[Real],
    t: Real,
    k: Real,
    method: FunctionType = crout,
):
    """
    compare with parabolic_forward method, parabolic_backward method is
    unconditional stable, which means we can choose a larger step so that
    we can reduce computation
    """
    assert alpha >= 0
    init: ndarray = np.asarray(init, dtype=np.float_)
    n: int = init.shape[0]
    assert n > 3
    assert k > 0
    x0, x1 = min(x0, x1), max(x0, x1)
    h: Real = (x1 - x0) / (n - 1)
    lbd: Real = alpha * k / h**2
    # setup iteration matrix
    a: ndarray = sparse.diags(
        (-lbd, 1 + 2 * lbd, -lbd), (-1, 0, 1), shape=(n - 2, n - 2)
    ).toarray()
    # considering it's a tridiagonal matrix,
    # we can use crout method to solve it
    y: ndarray = method(np.hstack((a, init[1:-1, np.newaxis])))
    return np.pad(y, (1, 1), constant_values=(bc[0](t), bc[1](t)))


def crank_nicolson(
    bc: Sequence[FunctionType, FunctionType],
    alpha: Real,
    x0: Real,
    x1: Real,
    init: Sequence[Real],
    t: Real,
    k: Real,
    method: FunctionType = crout,
):
    """
    compare with parabolic_backward method, which local trunc error is
    O(k + h^2). To increase the precision, Crank-Nicolson method is applied
    here with trunc error O(k^2 + h^2)
    """
    assert alpha >= 0
    init: ndarray = np.asarray(init, dtype=np.float_)
    n: int = init.shape[0]
    assert n > 3
    assert k > 0
    x0, x1 = min(x0, x1), max(x0, x1)
    h: Real = (x1 - x0) / (n - 1)
    lbd: Real = alpha * k / h**2
    # solving linear equation A*w_i+1 = B*w_i
    a: ndarray = sparse.diags(
        (-lbd / 2, 1 + lbd, -lbd / 2), (-1, 0, 1), shape=(n - 2, n - 2)
    ).toarray()
    b: ndarray = sparse.diags(
        (lbd / 2, 1 - lbd, lbd / 2), (-1, 0, 1), shape=(n - 2, n - 2)
    ).toarray()
    rhs: ndarray = b @ init[1:-1]
    # considering it's a tridiagonal matrix,
    # we can use crout method to solve it
    y: ndarray = method(np.hstack((a, rhs[:, np.newaxis])))
    return np.pad(y, (1, 1), constant_values=(bc[0](t), bc[1](t)))


def main():
    left = lambda t_: np.ones_like(t_)
    right = lambda t_: np.ones_like(t_)
    alpha = 1.0
    x0, x1 = 0.0, 5.0
    n = 20
    init = [1.0] * n
    k = 0.005

    m = 5
    c = "rgb"
    for i, method in enumerate([parabolic_forward, parabolic_backward, crank_nicolson]):
        data: ndarray = np.empty((m, n), dtype=np.float_)
        data[0] = method([left, right], alpha, x0, x1, init, 0, k)
        for j in range(1, m):
            data[j] = method([left, right], alpha, x0, x1, data[j - 1], 5 * j * k, k)
        plt.plot(data.T[0], "--", lw=1, c=c[i], label=method.__name__)
        plt.plot(data.T[1:], "--", lw=1, c=c[i])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
