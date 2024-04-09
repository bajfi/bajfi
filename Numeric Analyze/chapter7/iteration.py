from collections.abc import Sequence
from numbers import Real
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg, ndarray
from tqdm import trange

from chapter7.conjugate_gradient import conjugate_gradient


def _iter_precheck(
    a: Sequence[Sequence[Real]], b: Sequence[Real], x0: Real | Sequence[Real]
):
    if not isinstance(a, ndarray):
        a = np.asarray(a).squeeze()
    if a.ndim != 2 or not np.allclose(*a.shape):
        raise Exception("A must be square matrix")
    N = a.shape[0]
    if not isinstance(b, ndarray):
        b = np.asarray(b).squeeze()
    if b.ndim != 1:
        raise Exception("b should be 1d array")
    if N != b.shape[0]:
        raise Exception("A should be match with b")
    if isinstance(x0, Real):
        x0 = np.ones(b.shape[0], dtype=np.float_) * x0
    else:
        x0 = np.asarray(x0)
    if x0.shape != b.shape:
        raise Exception("x0 has unmatched shape")
    if (a.diagonal() == 0).any():
        raise Exception("diagonal entry can't be 0")
    return a, b, x0


def jacobi(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    tol: float = 1e-5,
    maximum_iter: int = 30,
    verbose=False,
    return_points: bool = False,
):
    """P453
    x = D^-1 * (L + U) * x' + D^-1 * b
    here we need all the entry on
    diagonal is non-zero
    """
    a, b, x0 = _iter_precheck(a, b, x0)
    N: int = a.shape[0]
    points: List[Real] = []
    for k in range(maximum_iter):
        if return_points:
            points.append(x0)
        x_k: ndarray = np.empty_like(x0, dtype=np.float_)
        for i in range(N):
            x_k[i] = (b[i] - np.dot(a[i], x0) + a[i][i] * x0[i]) / a[i][i]
        # check if converged
        err: Real = linalg.norm(x_k - x0)
        if verbose:
            print(f"Iteration {k + 1}:\n", x_k, f"\nerr: {err}")
        if err < tol:
            if return_points:
                points.append(x_k)
                return np.asarray(points)
            return np.asarray(x_k)
        x0: ndarray = x_k
    else:
        raise Exception("Maximum iteration exceed,\nMake sure a is diagonal dominant")


def sor(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    tol: float = 1e-5,
    maximum_iter: int = 30,
    omega=1.25,
    verbose=False,
    return_points: bool = False,
    **kwargs,
):
    """P467
    x = (1-w)XO + w * (D-L)^-1 * U * x + (D-L)^-1 * b
    here we need all the entry on
    diagonal is non-zero
    """
    assert 0 < omega < 2
    a, b, x0 = _iter_precheck(a, b, x0)
    N: int = a.shape[0]
    l: ndarray = np.tril(a, -1)
    u: ndarray = np.triu(a, 1)
    points: List = []
    for k in range(maximum_iter):
        if return_points:
            points.append(x0)
        x_k: ndarray = np.empty_like(x0, dtype=np.float_)
        for i in range(N):
            x_k[i] = (1 - omega) * x0[i] + omega * (
                b[i] - l[i][:i] @ x_k[:i] - u[i][i + 1 :] @ x0[i + 1 :]
            ) / a[i][i]
        err: Real = linalg.norm(x_k - x0)
        if verbose:
            print(f"Iteration {k + 1}:\n", x_k, f"\nerr: {err}")
        if err < tol:
            if return_points:
                points.append(x_k)
                return np.asarray(points)
            return np.asarray(x_k)
        x0: ndarray = x_k
    else:
        raise Exception("Maximum iteration exceed.\nMake sure a is diagonal dominant")


def gaussian_seidel(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    tol: float = 1e-5,
    maximum_iter: int = 30,
    verbose=False,
    return_points: bool = False,
    **kwargs,
):
    """P456
    x = (D-L)^-1 * U * x + (D-L)^-1 * b
    here we need all the entry on
    diagonal is non-zero
    Gaussian Seidel is a special kind of SOR
    """
    return sor(
        a,
        b,
        x0,
        tol,
        maximum_iter,
        omega=1,
        verbose=verbose,
        return_points=return_points,
        **kwargs,
    )


def ssor(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    tol: float = 1e-5,
    maximum_iter: int = 30,
    omega=1.25,
    verbose=False,
    return_points: bool = False,
):
    """
    Young, David M. (May 1, 1950),
    Iterative methods for solving partial difference equations of elliptical,
    PhD thesis, Harvard University, retrieved 2009-06-15
    """
    assert 0 < omega < 2
    a, b, x0 = _iter_precheck(a, b, x0)
    l = -np.tril(a, -1)
    u = -np.triu(a, 1)
    diag = np.diag(np.diag(a))
    points = []
    for iters in range(maximum_iter):
        if return_points:
            points.append(x0)
        x = linalg.inv(diag - omega * l) @ (
            ((1 - omega) * diag + omega * u) @ x0 + omega * b
        )
        x = linalg.inv(diag - omega * u) @ (
            ((1 - omega) * diag + omega * l) @ x + omega * b
        )
        err: Real = linalg.norm(x - x0)
        if verbose:
            print(f"Iteration {iters + 1}:\n", x, f"\nerr: {err}")
        if err < tol:
            if return_points:
                points.append(x)
                return np.asarray(points)
            return np.asarray(x0)
        x0 = x
    else:
        raise Exception("Maximum iteration exceed.\nMake sure a is diagonal dominant")


def refinement(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    precondition: Sequence[Sequence[Real]] = None,
    tol: float = 1e-5,
    maximum_iter: int = 30,
    omega=1.25,
    verbose=False,
    return_points: bool = False,
):
    """
    p474
    """
    return


def main():
    methods = [gaussian_seidel, jacobi, ssor, conjugate_gradient]
    iterTimes = 50
    size = 300
    errs = [[] for _ in range(len(methods))]
    iter_count = [[] for _ in range(len(methods))]
    cond = []

    fig, axes = plt.subplots(
        2, 1, sharex="col", figsize=(12, 8), constrained_layout=True
    )
    plt.suptitle(
        f"Compare between different iteration methods\n"
        f"Test on {size} x {size} symmetry diagonal dominant matrix"
    )

    for _ in trange(iterTimes):
        a = np.random.random((size, size))
        a = (a.T + a) / 2
        a += np.eye(size) * size
        b = (np.random.random(size) - 0.5) * size
        precondition = np.identity(a.shape[0]) / np.diag(a) ** 0.5
        for i, method in enumerate(methods):
            ans = method(a, b, 0, precondition=precondition, maximum_iter=100)
            errs[i].append(linalg.norm((a @ ans) - b))
        cond.append(linalg.cond(a))
    for i, method in enumerate(methods):
        axes[0].plot(range(iterTimes), errs[i], "o--", ms=2, label=method.__name__)

    axes[0].set_ylabel("Norm of residual vector")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot(range(iterTimes), cond, "o--", ms=2, label="condition number")
    axes[1].set_xlabel("No.")
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
