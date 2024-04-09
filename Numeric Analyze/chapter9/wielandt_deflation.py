from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import ndarray

from chapter9.power_method import power_method


def wielandt_deflation(
    A: Sequence[Sequence[Real]],
    eigen_value: Real,
    eigen_vector: Sequence[Real],
    power_method: FunctionType = power_method,
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = True,
):
    """
    p589
    we can use wielandt_deflation to find second dormant eigen value
    the input eigen value and vector can be got from inverse-power-method
    if all eigen values needed, similarity transform method can be applied
    """
    assert eigen_value != 0
    A: ndarray = np.asarray(A, dtype=np.float_)
    eigen_vector: ndarray = np.asarray(eigen_vector, dtype=np.float_)
    idx: int = np.argmax(np.abs(eigen_vector))
    eigen_vector /= eigen_vector[idx]
    x: ndarray = A[idx] / eigen_value
    B: ndarray = A - eigen_value * np.outer(eigen_vector, x)
    B: ndarray = B[1:, 1:]
    # use power method to get the second dormant eigen vector
    eig, eig_vec = power_method(
        B, np.ones(B.shape[0], dtype=np.float_), tol, maxIter, verbose
    )
    eig_vec = np.pad(eig_vec, (A.shape[0] - eig_vec.shape[0], 0), constant_values=0.0)
    # use eigen vector to calculate eigen value
    # accoring to formula 9.6 in p587
    eig_vec = (eig - eigen_value) * eig_vec + eigen_value * np.dot(
        A[idx], eig_vec
    ) * eigen_vector
    return eig, eig_vec


def main():
    A = np.array([[4.0, -1, 1], [-1, 3, -2], [1, -2, 3]])
    eig = 6.0
    eig_v = [1, -1, 1]
    eig2, eig_v2 = wielandt_deflation(A, eig, eig_v, verbose=True)
    print(eig2)
    print(eig_v2)


if __name__ == "__main__":
    main()
