from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import ndarray


def nonlinear_finite_difference(
    f: Sequence[FunctionType],
    f_y: Sequence[FunctionType],
    f_y1: Sequence[FunctionType],
    a: Real,
    b: Real,
    alpha: Real,
    beta: Real,
    N: int,
):
    """
    p693
    for non-linear problem, we can't determinate the coefficient matrix,
    which means we can not get the result by solving linear equations.
    So iterating method is chosen for this kind of problem
    """
    assert N > 1
    a, b = min(a, b), max(a, b)
    # make initial guess of the answer, which is a straight line
    w: ndarray = np.linspace(a, b, N + 2)

    return


def main():
    f1 = lambda x_, y0_, y1_: y1_
    f2 = lambda x_, y0_, y1_: 1 / 8 * (32 + 2 * x_**3 - y0_ * y1_)
    f_exact = lambda x_: x_**2 + 16 / x_
    a, b = 1.0, 2.0
    alpha: Real = 17.0
    beta: Real = 12.0


if __name__ == "__main__":
    main()
