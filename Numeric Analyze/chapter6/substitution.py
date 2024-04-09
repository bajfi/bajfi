import numpy as np


def backward_substitution(triuMatrix: np.ndarray):
    N = triuMatrix.shape[0]
    x = np.zeros(N, dtype=np.float_)
    x[-1] = triuMatrix[-1][-1] / triuMatrix[-1][-2]
    for i in range(N - 2, -1, -1):
        x[i] = (
            triuMatrix[i][-1]
            - np.sum([triuMatrix[i][j] * x[j] for j in range(N - 1, i, -1)])
        ) / triuMatrix[i][i]
    return x


def forward_substitution(trilMatrix: np.ndarray):
    N = trilMatrix.shape[0]
    x = np.zeros(N, dtype=np.float_)
    x[0] = trilMatrix[0][-1] / trilMatrix[0][0]
    for i in range(1, N):
        x[i] = (
            trilMatrix[i][-1] - np.sum([trilMatrix[i][j] * x[j] for j in range(i)])
        ) / trilMatrix[i][i]
    return x
