import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from tqdm.auto import trange

from chapter6.substitution import backward_substitution


def pivot_normal(augMatrix: np.ndarray):
    N = augMatrix.shape[0]
    # find first non-zero element
    # and swap two rows
    for i in range(N):
        if augMatrix[i][i] == 0:
            for j in range(i + 1, N):
                if augMatrix[j][i] != 0:
                    augMatrix[[i, j]] = augMatrix[[j, i]]
                    break
            # if all zeros, then terminate programe
            else:
                raise Exception("No unique solution")
    return augMatrix


def pivot_partial(augMatrix: np.ndarray):
    # partial pivoting
    # find the maximum in the rest of the column
    # then swap
    N = augMatrix.shape[0]
    for i in range(N):
        cur, row = np.abs(augMatrix[i][i]), i
        for j in range(i + 1, N):
            if np.abs(augMatrix[j][i]) > cur:
                cur = np.abs(augMatrix[i][j])
                row = j
        if cur == 0:
            raise Exception("No unique solution")
        augMatrix[[i, row]] = augMatrix[[row, i]]  # row exchange
    return augMatrix


def pivot_scale(augMatrix: np.ndarray):
    # scale partial pivoting
    # get maximum of each row s_i
    # find the row with maximum relevant in this row
    # then swap
    N = augMatrix.shape[0]
    s = np.max(np.abs(augMatrix[:, :-1]), axis=1)
    if np.all(s):
        for i in range(N):
            cur, row = np.abs(augMatrix[i, i]) / s[i], i
            for j in range(i + 1, N):
                rate = np.abs(augMatrix[j][i]) / s[j]
                if rate > cur:
                    cur = rate
                    row = j
            augMatrix[[i, row]] = augMatrix[[row, i]]
            s[i], s[row] = s[row], s[i]  # this is important
    else:  # the row are all zero
        raise Exception("No unique solution")
    return augMatrix


def guass_eliminate(augMatrix: np.ndarray, pivot=""):
    if augMatrix.ndim != 2:
        raise Exception("input matrix has wrong shape")
    N = augMatrix.shape[0]
    pivot = pivot.lower()
    if not pivot:
        pivot_normal(augMatrix)
    elif pivot.startswith("s"):
        pivot_scale(augMatrix)
    elif pivot.startswith("p"):
        pivot_partial(augMatrix)
    else:
        raise NotImplementedError
    for i in range(N - 1):
        # backward substitution
        for j in range(i + 1, N):
            m_ji = augMatrix[j][i] / augMatrix[i][i]
            augMatrix[j] -= m_ji * augMatrix[i]
    # if a[n][n] == 0, means no solution
    if augMatrix[N - 1][N - 1] == 0:
        raise Exception("No solution")
    return augMatrix


# mat = np.array([[2.11, -4.21, 0.92, 2.01],
#                 [4.01, 10.2, -1.12, -3.09],
#                 [1.09, 0.987, 0.832, 4.21]])
# pivot_scale(mat)
# mat_g = guassEliminate(np.array(mat), 's')
# print(mat)
# ans = backSubstitution(mat_g)
# print(mat[:, :-1] @ ans)


def main():
    fig, axes = plt.subplots(
        2, 1, sharex="col", figsize=(12, 8), constrained_layout=True
    )
    error = [[], [], []]
    times = [[], [], []]
    N = 200
    Iter = 50
    bound = 100000
    for i in trange(Iter):
        augMatrix = (np.random.random((N, N + 1)) - 0.5) * 2 * bound
        ans = (np.random.random(N) - 0.5) * 2 * bound
        augMatrix[:, -1] = augMatrix[:, :-1] @ ans
        for j, pivot in enumerate(["", "p", "s"]):
            mat = np.array(augMatrix)
            start = time.time()
            guass_eliminate(mat, pivot)
            x = backward_substitution(mat)
            times[j].append(time.time() - start)
            err = linalg.norm(x - ans)
            error[j].append(err)
    xx = np.arange(Iter)
    axes[0].plot(xx, error[0], label="normal")
    axes[0].plot(xx, error[1], label="partial pivot")
    axes[0].plot(xx, error[2], label="scale partial")
    axes[1].plot(xx, times[0], label="normal")
    axes[1].plot(xx, times[1], label="partial pivot")
    axes[1].plot(xx, times[2], label="scale partial")
    axes[0].legend()
    axes[1].legend()
    plt.suptitle("Compare on different pivot methods", fontdict={"size": 16})
    axes[0].set_title(f"Test on {N} $\\times$ {N} matrix")
    axes[0].set_ylabel("Error")
    axes[1].set_ylabel("time cost $(s)$")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    plt.xlabel("Test times")
    plt.show()


if __name__ == "__main__":
    main()
