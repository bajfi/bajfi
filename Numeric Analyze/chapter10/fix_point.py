from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import linalg, ndarray


def fix_point(
    f: Sequence[FunctionType],
    x0: Sequence[Real],
    tol: Real = 1e-5,
    maxIter: int = 30,
    verbose: bool = False,
):
    """
    p630
    this method is used to find the root of coordinate function
    with strict limitations, make sure the functions satisfy
    those conditions before iterating

    x0 should be chosen with great consideration, or something
    like complex root will occur unexpectedly
    """
    N: int = len(f)
    assert N == len(x0)
    x0: ndarray = np.asarray(x0, dtype=np.float_)
    for iter_ in range(maxIter):
        # x1: ndarray = np.asarray([f[i](*x0) for i in range(len(f))])
        # use Gaussian Seidel method can accelerate the process
        x1: ndarray = np.array(x0)
        x1[0] = f[0](*x0)
        for i in range(1, N):
            x1[i] = f[i](*x1)
        err: Real = linalg.norm(x0 - x1)
        if verbose:
            print(
                f"Iteration {iter_ + 1}\nx1:\n",
                x1,
                "\n||x0 - x1||:\n",
                f"\nIteration {iter_ + 1} \n",
                err,
            )
        if err < tol:
            return x1
        x0 = x1
    else:
        raise Exception("maximum iteration exceeded !!")


def main():
    f1 = lambda x1_, x2_, x3_: 1 - np.cos(x1_ * x2_ * x3_)
    f2 = lambda x1_, x2_, x3_: 1 - (1 - x1_) ** 0.25 - 0.5 * x3_**2 + 0.15 * x3_
    f3 = lambda x1_, x2_, x3_: x1_**2 + 0.1 * x2_**2 - 0.01 * x2_ + 1

    x = fix_point([f1, f2, f3], [0.1, 0.1, 1.0], verbose=True, tol=1e-12)


if __name__ == "__main__":
    main()
