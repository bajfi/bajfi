from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy import ndarray
from scipy import linalg


def house_holder(A: Sequence[Sequence[Real]], verbose: bool = False):
    """
    p600
    the A matrix can be arbitrary,
    if non symmetry matrix is given
    then return an upper Hessenberg matrix
    """
    A: ndarray = np.asarray(A, dtype=np.float_)
    N: int = A.shape[0]
    I: ndarray = np.identity(N, dtype=np.float_)
    w: ndarray = np.empty(N, dtype=np.float_)
    for i in range(N - 2):
        alpha: Real = -np.sign(A[i + 1][i]) * linalg.norm(A[i + 1 :, i])
        r = np.sqrt(0.5 * alpha**2 - 0.5 * A[i + 1][i] * alpha)
        w[i] = 0
        w[i + 1] = 0.5 * (A[i + 1][i] - alpha) / r
        for j in range(i + 2, N):
            w[j] = 0.5 * A[j][i] / r
        P: ndarray = I - 2 * np.outer(w, w)
        A = P @ A @ P
        if verbose:
            print(f"P_{i + 1}\n {P.round(2)}")
            print(f"A_{i + 2}\n {A.round(2)}")
    return A


def main():
    A = np.array([[4.0, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [3, 1, -2, -1]])

    house_holder(A, True)


if __name__ == "__main__":
    main()
