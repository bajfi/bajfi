from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy import sparse

from chapter7.conjugate_gradient import conjugate_gradient


def poisson(
    f: FunctionType,
    bc: Sequence[FunctionType],
    x0: Real,
    x1: Real,
    y0: Real,
    y1: Real,
    m: int,
    n: int,
    method: FunctionType = conjugate_gradient,
    init: Sequence[Real] | Real = 0,
    precondition: Sequence[Sequence[Real]] = None,
    tol: float = 1e-5,
    maximum_iter: int = 30,
    verbose=False,
    **kwargs
):
    """
    p720
    Use finite different method to solve poisson function
    here the labeling order of points is from bottom-left to top-right

    :param f: function on the rhs
    :param bc: boundary condition functions, default order is bottom, up, left and right
    :param m: sub-intervals in x-axis
    :param n: sub-intervals in y-axis
    :param method: method to solve linear equations, for small size matrix,
    gaussian elimination method can be used
    :param init: initial guess for gaussian-seidel method
    """
    assert m > 1
    assert n > 1
    assert len(bc) == 4
    N: int = m * n
    x0, x1 = min(x0, x1), max(y0, y1)
    y0, y1 = min(y0, y1), max(y0, y1)
    xx: ndarray = np.linspace(x0, x1, m + 2)
    yy: ndarray = np.linspace(y0, y1, n + 2)
    # calculate boundary value
    bottom: ndarray = bc[0](xx)
    up: ndarray = bc[1](xx)
    left: ndarray = bc[2](yy)
    right: ndarray = bc[3](yy)
    # setup interation matrix
    diag_1: ndarray = -np.ones(N, dtype=np.float_)
    diag_1[m - 1 :: m] = 0.0
    a: ndarray = sparse.diags(
        (4, -1, -1, diag_1, diag_1), offsets=(0, m, -m, 1, -1), shape=(N, N)
    ).toarray()
    # calculate value of b
    b: ndarray = -f(*np.meshgrid(xx[1:-1], yy[1:-1])).ravel()
    b[:m] = bottom[1:-1]
    b[-m:] = up[1:-1]
    b[::m] += left[1:-1]
    b[m - 1 :: m] += right[1:-1]
    # solving equations
    y: ndarray = method(
        a,
        b,
        init,
        tol=tol,
        maximum_iter=maximum_iter,
        verbose=verbose,
        precondition=precondition,
        **kwargs
    ).reshape((m, n))
    return y


def main():
    f = lambda x_, y_: x_ * np.exp(y_)
    bottom = lambda x_: x_
    up = lambda x_: np.exp(x_) * x_
    left = lambda y_: np.zeros_like(y_, dtype=np.float_)
    right = lambda y_: 2.0 * np.exp(y_)
    x0, x1 = 0.0, 2.0
    y0, y1 = 0.0, 2.0
    m, n = 50, 50
    xx = np.linspace(x0, x1, m + 2)
    yy = np.linspace(y0, y1, n + 2)

    a = poisson(
        f,
        [bottom, up, left, right],
        x0,
        x1,
        y0,
        y1,
        m=m,
        n=n,
        maximum_iter=500,
        verbose=True,
        tol=1e-3,
        precondition=np.identity(m * n) * 0.5,
    )

    # show data
    data: DataFrame = DataFrame(a, index=yy[1:-1].round(2), columns=xx[1:-1].round(2))
    sns.heatmap(data, annot=False, fmt=".1f", cmap="viridis")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()
