import numpy as np
from numpy import linalg, ndarray


def lu(a: np.ndarray):
    """P406"""
    if a.ndim != 2 or not np.allclose(*a.shape):
        raise Exception("matrix mush be square")
    if a[0][0] == 0:
        raise Exception("Factorization impossible")
    N: int = a.shape[0]
    L: ndarray = np.eye(N, dtype=np.float_)  # diagonal of L is 1
    U: ndarray = np.eye(N, dtype=np.float_)
    # step 2
    L[:, 0] = a[:, 0]
    U[0, 1:] = a[0, 1:] / L[0][0]
    # step 3
    for i in range(1, N - 1):
        # step 4
        L[i][i] = a[i][i] - np.sum([L[i][k] * U[k][i] for k in range(i)])
        if np.allclose(U[i][i], 0):
            raise Exception("Factorization impossible")
        # step 5
        for j in range(i + 1, N):
            U[i][j] = (a[i][j] - np.sum([L[i][k] * U[k][j] for k in range(i)])) / L[i][
                i
            ]
            L[j][i] = a[j][i] - np.sum([L[j][k] * U[k][i] for k in range(i)])
    # step 6
    L[-1][-1] = a[-1][-1] - np.sum([L[-1][k] * U[k][-1] for k in range(N - 1)])
    return L, U


def ldl(matrix: np.ndarray):
    """P417"""
    if matrix.ndim != 2 or not np.allclose(*matrix.shape):
        raise Exception("matrix mush be square")
    N: int = matrix.shape[0]
    L: ndarray = np.eye(N, dtype=np.float_)
    D: ndarray = np.ones(N, dtype=np.float_)
    for i in range(N):
        V: ndarray = np.zeros(N, dtype=np.float_)
        for j in range(i):
            V[j] = L[i][j] * D[j]
        D[i] = matrix[i][i] - np.sum([L[i][j] * V[j] for j in range(i)])
        if np.allclose(D[i], 0):
            raise Exception("Entry on diagonal of D must be non-zero")
        for j in range(i, N):
            L[j][i] = (matrix[j][i] - np.sum([L[j][k] * V[k] for k in range(i)])) / D[i]
    return L, D


def cholesky(a: np.ndarray):
    """P419"""
    if a.ndim != 2 or not np.allclose(*a.shape):
        raise Exception("matrix mush be square")
    N: int = a.shape[0]
    L: ndarray = np.eye(N, dtype=np.float_)
    L[0, 0] = np.sqrt(a[0][0])
    L[1:, 0] = a[1:, 0] / L[0][0]
    for i in range(1, N - 1):
        L[i][i] = np.sqrt(a[i][i] - linalg.norm(L[i][:i]) ** 2)
        for j in range(i + 1, N):
            L[j][i] = (a[j][i] - np.sum([L[j][k] * L[i][k] for k in range(i)])) / L[i][
                i
            ]
    L[-1][-1] = np.sqrt(a[-1][-1] - linalg.norm(L[-1][: N - 1]) ** 2)
    return L


def crout(a: np.ndarray):
    """P422"""
    if a.ndim != 2 or a.shape[0] + 1 != a.shape[1]:
        raise Exception("a should be an augmented matrix")
    N: int = a.shape[0]
    L: ndarray = np.ones((2, N), dtype=np.float_)  # first row is diagonal
    U: ndarray = np.ones(N, dtype=np.float_)  # diagonal are 1
    Z: ndarray = np.ones(N, dtype=np.float_)
    x: ndarray = np.ones_like(Z)
    # step 1-3 set up to solve Lz = b
    # step 1
    L[0][0] = a[0][0]
    U[0] = a[0][1] / L[0][0]
    Z[0] = a[0][-1] / L[0][0]
    # step 2
    for i in range(1, N - 1):
        L[1][i - 1] = a[i][i - 1]
        L[0][i] = a[i][i] - L[1][i - 1] * U[i - 1]
        U[i] = a[i][i + 1] / L[0][i]
        Z[i] = (a[i][-1] - L[1][i - 1] * Z[i - 1]) / L[0][i]
    # step 3
    L[1][-2] = a[-1][-3]
    L[0][-1] = a[-1][-2] - L[1][-2] * U[-2]
    Z[-1] = (a[-1][-1] - L[1][-2] * Z[-2]) / L[0][-1]
    # step 4-5 solve Ux = z
    x[-1] = Z[-1]
    for i in range(N - 2, -1, -1):
        x[i] = Z[i] - U[i] * x[i + 1]
    return x
