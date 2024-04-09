import warnings
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg, ndarray
from scipy import sparse

from chapter6.factorization import crout


def linear_finite_difference(
    px: FunctionType,
    qx: FunctionType,
    rx: FunctionType,
    a: Real,
    b: Real,
    alpha: Real,
    beta: Real,
    N: int,
):
    """
    p687
    To deal with the derivative terms, center difference method is use
    here with trunc error O(h^2). After that, the ODE function becomes
    a tridiagonal matrix equation, which can be solved with crout method
    implemented in chapter 6
    """
    assert N > 1
    a, b = min(a, b), max(a, b)
    h: Real = (b - a) / (N + 1)
    x: ndarray = np.linspace(a, b, N + 2)
    # according to theorem 11.3, q(x) must be continuous in [a,b] and q(x) > 0
    # with assumption h < L/2, where L = max(|p(x)|)
    if np.any(qx(x) < 0):
        warnings.warn("q(x) must be continuous in [a,b] and q(x) >= 0")
    if h >= linalg.norm(px(x), 1) / 2:
        raise ValueError("h < L/2, where L = max(|p(x)|) with x in [a,b]")
    # setup the discrete matrix
    # if sparse method is complemented, use that
    low_diag: ndarray = -1.0 - px(x[2:-1]) * h / 2
    mid_diag: ndarray = 2.0 + h**2 * qx(x[1:-1])
    up_diag: ndarray = -1.0 + px(x[1:-2]) * h / 2
    mat: ndarray = sparse.diags((low_diag, mid_diag, up_diag), (-1, 0, 1)).toarray()
    b: ndarray = -(h**2) * rx(x[1:-1])
    b[0] += (1 + h / 2 * px(x[1])) * alpha
    b[-1] += (1 - h / 2 * px(x[N])) * beta
    # combine mat and b to get augment matrix
    mat: ndarray = np.hstack((mat, b[:, np.newaxis]))
    # solve matrix with crout method
    y: ndarray = crout(mat)
    return np.pad(y, (1, 1), "constant", constant_values=(alpha, beta))


def main():
    fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 8))
    px = lambda x_: 2.0 * np.ones_like(x_)
    qx = lambda x_: -1.0 * np.ones_like(x_)
    rx = lambda x_: x_ * np.exp(x_) - x_
    a, b = 0.0, 2.0
    alpha = 0.0
    beta = -4.0
    N = 19
    h = (b - a) / (N + 1)
    x = np.linspace(a, b, N + 2)

    y_exact = 1 / 6 * x**3 * np.exp(x) - 5 / 3 * x * np.exp(x) + 2 * np.exp(x) - x - 2
    axes[0].plot(x, y_exact, "--", label="exact value")

    y_0 = linear_finite_difference(px, qx, rx, a, b, alpha, beta, N)
    axes[0].plot(x, y_0, "o", ms=4, label=f"h = {h}")
    err_0 = np.abs(y_0 - y_exact)
    axes[1].plot(x, err_0, "o", ms=4, label=f"h = {h}")

    axes[1].set_yscale("log")
    axes[0].grid(axis="y")
    axes[0].legend()
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
