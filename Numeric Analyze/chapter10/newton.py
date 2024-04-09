from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import linalg, ndarray

from chapter6.gaussian_elimination import guass_eliminate
from chapter6.substitution import backward_substitution


def newton(
    f: Sequence[FunctionType],
    jacobian: FunctionType,
    x0: Sequence[Real],
    pivot: str = "partial",
    tol: Real = 1e-5,
    maxIter: int = 30,
    verbose: bool = False,
):
    """
    p641
    for some problems difficult to determinate the representation of
    each variable, newton method is a good choice
    """
    N: int = len(f)
    assert N == len(x0)
    x0: ndarray = np.asarray(x0, dtype=np.float_)
    # get Jacobian matrix with initial guess
    # considering the iterating process, x_k+1 = x_k - J^-1 * F
    # explicit computation of J^-1 should be avoid, by perform two steps
    # first solve linear equations J*Y = -F
    # then x_k+1 = x_k + Y
    for i in range(maxIter):
        Jacobian: Sequence[Sequence[Real]] = jacobian(*x0)
        neg_Fx: Sequence[Sequence[Real]] = [[-f[i](*x0)] for i in range(N)]
        # partial pivot technique is set as default
        augMatrix: ndarray = np.hstack((Jacobian, neg_Fx))
        guass_eliminate(augMatrix, pivot)
        y: ndarray = backward_substitution(augMatrix)
        x1: ndarray = x0 + y
        err: Real = linalg.norm(y)
        if verbose:
            print(f"Iteration {i + 1}\nx1:\n", x1, "\n||x0 - x1||:\n", err)
        if err < tol:
            return x1
        x0 = x1
    else:
        raise Exception("maximum iteration exceeded !!")


def broyden(
    f: Sequence[FunctionType],
    jacobian: FunctionType,
    x0: Sequence[Real],
    tol: Real = 1e-5,
    maxIter: int = 30,
    verbose: bool = False,
):
    """
    p650
    the Broyden method avoid calculating Jacobian matrix and solving
    linear equations every iteration, which reduce computation but lose
    some precision
    """
    N: int = len(f)
    assert N == len(x0)
    # iteration setup
    fx: ndarray = np.asarray([f[i](*x0) for i in range(N)], dtype=np.float_)
    x0: ndarray = np.asarray(x0, dtype=np.float_)
    A: ndarray = np.asarray(jacobian(*x0), dtype=np.float_)  # A = jacobian matrix
    A_inv: ndarray = linalg.inv(A)
    s: ndarray = -A_inv @ fx
    x0 += s
    if verbose:
        print(f"Iteration 1\nx:\n", x0, "\n||x1 - x0||:\n", linalg.norm(s))
    # iteration process
    for i in range(1, maxIter):
        fx1: ndarray = np.asarray([f[i](*x0) for i in range(N)])
        fx_diff: ndarray = fx1 - fx
        z: ndarray = -A_inv @ fx_diff
        p: Real = -np.dot(s, z)
        u_t: ndarray = A_inv.T @ s
        A_inv += np.outer(s + z, u_t) / p
        s: ndarray = -A_inv @ fx1
        x0 += s
        err: Real = linalg.norm(s)
        fx = fx1  # update fx
        if verbose:
            print(f"Iteration {i + 1}\nx1:\n", x0, "\n||x0 - x1||:\n", err)
        if err < tol:
            return x0
    else:
        raise Exception("maximum iteration exceeded !!")


def main():
    # f1 = lambda x1_, x2_, x3_: 3 * x1_ - np.cos(x2_ * x3_) - 0.5
    # f2 = lambda x1_, x2_, x3_: x1_ ** 2 - 81 * (x2_ + 0.1) ** 2 + np.sin(x3_) + 1.06
    # f3 = lambda x1_, x2_, x3_: np.exp(-x1_ * x2_) + 20 * x3_ + (10 * np.pi - 3) / 3
    #
    # jacobian = lambda x1_, x2_, x3_: [
    #     [3.0, x3_ * np.sin(x2_ * x3_), x2_ * np.sin(x2_ * x3_)],
    #     [2 * x1_, -162 * (x2_ + 0.1), np.cos(x3_)],
    #     [-x2_ * np.exp(-x1_ * x2_), -x1_ * np.exp(-x1_ * x2_), 20],
    # ]

    f1 = lambda x1_, x2_, x3_: 2 * x1_ - 3 * x2_ + x3_ - 4
    f2 = lambda x1_, x2_, x3_: 2 * x1_ + x2_ - x3_ + 4
    f3 = lambda x1_, x2_, x3_: x1_**2 + x2_**2 + x3_**2 - 4

    jacobian = lambda x1_, x2_, x3_: [
        [2.0, -3, 1],
        [2, 1, -1],
        [2 * x1_, 2 * x2_, 2 * x3_],
    ]

    # newton([f1, f2, f3], jacobian, [0.1, 0.1, -0.1], verbose=True)
    broyden([f1, f2, f3], jacobian, [0.1, 0.1, -0.1], verbose=True)


if __name__ == "__main__":
    main()
