from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from chapter4.composite import composite_simpson
from chapter6.gaussian_elimination import guass_eliminate
from chapter6.substitution import backward_substitution


def rational_approximation(poly_coef: Sequence[Real], p: int, q: int, pivot="partial"):
    """
    p531
    the coefficient should in degree increasing order
    default pivot is partial pivot
    """
    poly_coef: ndarray = np.asarray(poly_coef)
    # degree of r is equal to M(degree(p)) + N(degree(q))
    r_degree: int = p + q
    assert len(poly_coef) > r_degree
    # build augment matrix
    mat: ndarray = np.zeros((r_degree, r_degree + 1), dtype=np.float_)
    for i in range(r_degree):
        mat[i][-1] = -poly_coef[i + 1]
        mat[i][i] -= i <= p - 1
        for j in range(min(i + 1, q)):
            mat[i][p + j] = poly_coef[i - j]
    guass_eliminate(mat, pivot)
    # ans = [p0,p1,...pm,q1,q2,...qn]
    coefs: ndarray = guass_eliminate(mat)
    # return the polynomial ratio result
    # q0 is 1 as default, p0 = a0
    fp = lambda x: poly_coef[0] + sum(coefs[i] * x ** (i + 1) for i in range(p))
    fq = lambda x: 1 + sum(coefs[i + p] * x ** (i + 1) for i in range(q))
    return lambda x: fp(x) / fq(x)


def chebyshev_approximation(
    f: FunctionType,
    p: int,
    q: int,
    integral_method: FunctionType = composite_simpson,
    integral_step: Real = 1e-3,
    pivot: str = "partial",
):
    """
    p534
    the coefficient should in degree increasing order
    default pivot is partial pivot
    """
    # degree of r is equal to degree(p) + Ndegree(q)
    r_degree: int = p + q
    # integral to get lhs coefficient
    a: ndarray = np.empty(r_degree + q + 1, dtype=np.float_)
    # doubled a0 for computational efficiency
    for i in range(a.shape[0]):
        a[i] = (
            integral_method(
                lambda x: f(np.cos(x)) * np.cos(i * x),
                0,
                np.pi,
                round(np.pi / integral_step),
            )
            * 2
            / np.pi
        )
    # set up augMatrix
    mat: ndarray = np.zeros((r_degree + 1, r_degree + 2), dtype=np.float_)
    for i in range(mat.shape[0]):
        mat[i][i] = i <= p
        for j in range(p + 1, r_degree + 1):
            mat[i][j] = -0.5 * a[i + j - p] - (i > 0) * 0.5 * a[abs(i - j + p)]
        mat[i][-1] = 0.5 ** (i <= 0) * a[i]

    # solving the equations by gaussian elimination with pivot
    guass_eliminate(mat, pivot)
    # ans = [p0,p1,...pm,q1,q2,...qn]
    coefs: ndarray = backward_substitution(mat)
    # return the polynomial ratio result
    # q0 is 1 as default
    chebyshev_poly = [
        lambda x: 1.0,
        lambda x: x,
        lambda x: 2 * x**2 - 1.0,
        lambda x: 4 * x**3 - 3 * x,
        lambda x: 8 * x**4 - 8 * x**2 + 1,
        lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        lambda x: 32 * x**6 - 48 * x**4 + 18 * x**2 - 1,
        lambda x: 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x,
        lambda x: 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**4 + 1,
        lambda x: 256 * x**9 - 576 * x**7 + 432 * x**5 - 120 * x**3 + 9 * x,
        lambda x: 512 * x**10
        - 1280 * x**8
        + 1120 * x**6
        - 400 * x**4
        + 50 * x**2
        - 1,
        lambda x: 1024 * x**11
        - 2816 * x**9
        + 2816 * x**7
        - 1232 * x**5
        + 220 * x**3
        - 11 * x,
    ]
    fp = lambda x: sum(coefs[i] * chebyshev_poly[i](x) for i in range(p + 1))
    fq = lambda x: 1 + sum(
        coefs[i] * chebyshev_poly[i - p](x) for i in range(p + 1, r_degree + 1)
    )
    return lambda x: fp(x) / fq(x)


def fft():
    """
    p554
    """
    return


def main():
    fig, axes = plt.subplots(
        2, 1, sharex="col", figsize=(10, 8), constrained_layout=True
    )
    f_exact = lambda x: np.exp(-x)

    poly = [1.0, -1.0, 1.0 / 2, -1.0 / 6, 1.0 / 24, -1.0 / 120, 1.0 / 720]

    xx = np.linspace(0, 5)
    yy_exact = f_exact(xx)
    axes[0].plot(xx, f_exact(xx), "o--", ms=2, label="exact value")

    # we compare with Taylor series with degree 5
    f_taylor = lambda x: sum(n * x**i for i, n in enumerate(poly))
    yy_taylor = f_taylor(xx)
    axes[0].plot(xx, yy_taylor, "o--", ms=2, label="Taylor expansion")
    err_taylor = np.abs(yy_taylor - yy_exact)
    axes[1].plot(xx, err_taylor, "o--", ms=2, label="Taylor expansion - 5")

    for method in [rational_approximation]:
        f = method(poly, len(poly) // 2, (len(poly) - 1) // 2)
        yy = f(xx)
        axes[0].plot(xx, f(xx), "o--", ms=2, label=method.__name__)
        axes[1].plot(xx, np.abs(yy - yy_exact), "o--", ms=2, label=method.__name__)

    for method in [chebyshev_approximation]:
        f = method(f_exact, len(poly) // 2, (len(poly) - 1) // 2)
        yy = f(xx)
        axes[0].plot(xx, f(xx), "o--", ms=2, label=method.__name__)
        axes[1].plot(xx, np.abs(yy - yy_exact), "o--", ms=2, label=method.__name__)

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
