import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy import ndarray
from scipy import sparse


def rotation(R: ndarray) -> ndarray:
    """
    subroutine of QR process
    """
    N: int = R.shape[0]
    Q: ndarray = np.identity(N)
    for i in range(N - 1):
        cosine: Real = R[i, i] / math.hypot(R[i + 1, i], R[i, i])
        sine: Real = np.sqrt(1 - cosine**2)
        rotate_block = sparse.coo_array([[cosine, sine], [-sine, cosine]])
        rotate_matrix = sparse.block_diag(
            (
                sparse.identity(i, dtype=np.float_),
                rotate_block,
                sparse.identity(N - i - 2, dtype=np.float_),
            )
        )
        # update A matrix, R and Q
        R: ndarray = rotate_matrix @ R
        Q: ndarray = Q @ rotate_matrix.T
    return R @ Q


def qr(
    A_matrix: Sequence[Sequence[Real]],
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = False,
) -> ndarray:
    """
    qr process transforms a tridiagonal matrix A
    into a similar matrix which has been scale down
    """
    # make sure A is tridiagonal symmetry matrix
    R: ndarray = np.asarray(A_matrix, dtype=np.float_)
    assert R.ndim > 1
    tridiagonal = sparse.diags([np.diag(R, -1), np.diag(R), np.diag(R, 1)], [-1, 0, 1])
    assert np.allclose(R - tridiagonal, 0, atol=1e-12)
    # iteration until b_n is within telerance
    for i in range(maxIter):
        R = rotation(R)
        if verbose:
            print(f"Iteration {i + 1}\n", R.round(3))
        if np.all(abs(R[-2, -1]) < tol):
            return R
    else:
        raise Exception("maximum iteration exceeded !!")


def qr_refine(
    A_matrix: Sequence[Sequence[Real]],
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = False,
) -> ndarray:
    """
    the converge rate of original qr process is quite slow
    the shift technique is used to accelerate the process
    """
    # make sure A is tridiagonal symmetry matrix
    R: ndarray = np.asarray(A_matrix, dtype=np.float_)
    assert R.ndim > 1
    tridiagonal = sparse.diags([np.diag(R, -1), np.diag(R), np.diag(R, 1)], [-1, 0, 1])
    assert np.allclose(R - tridiagonal, 0, atol=1e-12)
    N: int = R.shape[0]
    I = sparse.identity(N)
    shift = 0
    # iteration until b_n is within telerance
    for i in range(maxIter):
        # get eigen values of sub-matrix
        a: Real = R[-1, -1] + R[-2, -2]
        b: Real = math.hypot(R[-1, -1] - R[-2, -2], 2 * R[-2, -1])
        lambda1, lambda2 = (a + b) / 2, (a - b) / 2
        # choose the value closer to A_nn
        shift_ = min(lambda1, lambda2, key=lambda x: abs(x - R[-1, -1]))
        shift += shift_
        R -= I * shift_
        R = rotation(R)
        if verbose:
            print(f"Iteration {i + 1}\n", (R + shift * I).round(3))
        if np.all(abs(R[-2, -1]) < tol):
            return R + I * shift
    else:
        raise Exception("maximum iteration exceeded !!")


def main():
    a = sparse.dia_matrix(
        (([2] * 5, [4] * 5, [2] * 5), (-1, 0, 1)), shape=(5, 5)
    ).toarray()
    # N = 7
    # dia0 = np.random.random(N) * N
    # dia1 = np.random.random(N) * N
    # a = sparse.dia_matrix(((np.roll(dia1, -1), dia0, dia1), (-1, 0, 1)),
    #                       shape=(N, N)).toarray()
    qr_refine(a, verbose=True)
    # qr(a, verbose=True)


if __name__ == "__main__":
    main()
