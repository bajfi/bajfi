from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import ndarray

from chapter6.gaussian_elimination import guass_eliminate
from chapter6.substitution import backward_substitution


def homotopy(
    f: Sequence[FunctionType],
    jacobian: FunctionType,
    x0: Sequence[Real],
    N: int,
    pivot: str = "partial",
    verbose: bool = False,
):
    """
    p666
    """
    length: int = len(f)
    assert length == len(x0)
    # iteration setup
    x0: ndarray = np.asarray(x0, dtype=np.float_)
    h: Real = 1 / N
    b: Sequence[Real] = [[-h * f[i](*x0)] for i in range(length)]
    # use Runge-Kutta method (Ralston-4) to iterate
    for i in range(N):
        A: Sequence[Sequence[Real]] = jacobian(*x0)
        aug_matrix: ndarray = np.hstack((A, b))
        guass_eliminate(aug_matrix, pivot)
        k1: ndarray = backward_substitution(aug_matrix)
        aug_matrix: ndarray = np.hstack((jacobian(*(x0 + 0.4 * k1)), b))
        guass_eliminate(aug_matrix, pivot)
        k2: ndarray = backward_substitution(aug_matrix)
        aug_matrix: ndarray = np.hstack(
            (jacobian(*(x0 + 0.29697761 * k1 + 0.15875964 * k2)), b)
        )
        guass_eliminate(aug_matrix, pivot)
        k3: ndarray = backward_substitution(aug_matrix)
        aug_matrix: ndarray = np.hstack(
            (jacobian(*(x0 + 0.21810040 * k1 - 3.05096516 * k2 + 3.83286476 * k3)), b)
        )
        guass_eliminate(aug_matrix, pivot)
        k4: ndarray = backward_substitution(aug_matrix)
        x0 += 0.17476028 * k1 - 0.55148066 * k2 + 1.20553560 * k3 + 0.17118478 * k4
        if verbose:
            print(f"Iteration {i + 1}\nx:\n", x0)
    return x0


def main():
    f1 = lambda x1_, x2_, x3_: 3 * x1_ - np.cos(x2_ * x3_) - 0.5
    f2 = lambda x1_, x2_, x3_: x1_**2 - 81 * (x2_ + 0.1) ** 2 + np.sin(x3_) + 1.06
    f3 = lambda x1_, x2_, x3_: np.exp(-x1_ * x2_) + 20 * x3_ + (10 * np.pi - 3) / 3

    jacobian = lambda x1_, x2_, x3_: [
        [3.0, x3_ * np.sin(x2_ * x3_), x2_ * np.sin(x2_ * x3_)],
        [2 * x1_, -162 * (x2_ + 0.1), np.cos(x3_)],
        [-x2_ * np.exp(-x1_ * x2_), -x1_ * np.exp(-x1_ * x2_), 20],
    ]

    homotopy([f1, f2, f3], jacobian, [0.0, 0.0, 0.0], 10, verbose=True)


if __name__ == "__main__":
    main()
