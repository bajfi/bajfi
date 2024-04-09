from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy import linalg, ndarray


def _precheck(
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
        x0: ndarray = np.ones(b.shape[0], dtype=np.float_) * x0
    else:
        x0: ndarray = np.asarray(x0)
    if x0.shape != b.shape:
        raise Exception("x0 has unmatched shape")
    if (a.diagonal() == 0).any():
        raise Exception("diagonal entry can't be 0")
    return a, b, x0


def conjugate_gradient(
    a: Sequence[Sequence[Real]],
    b: Sequence[Real],
    x0: Real | Sequence[Real],
    precondition: Sequence[Sequence[Real]] = None,
    tol: float = 1e-5,
    maximum_iter: int = 30,
    verbose=False,
    return_points: bool = False,
):
    """
    p488
    when precondiction matrix is I, the method is
    become non precondition
    """
    a, b, x0 = _precheck(a, b, x0)
    # handle precondition type
    if precondition is None:
        precondition: ndarray = np.identity(a.shape[0])
    if not isinstance(precondition, ndarray):
        precondition = np.asarray(precondition)
    # compute r0
    r: ndarray = b - a @ x0
    w: ndarray = precondition @ r
    v: ndarray = precondition @ w
    alpha = linalg.norm(w, 2) ** 2

    for iter_ in range(maximum_iter):
        if linalg.norm(v) < tol:
            return x0
        u: ndarray = a @ v
        t: Real = alpha / np.dot(u, v)
        x0 += t * v
        r -= t * u
        w: ndarray = precondition @ r
        beta: Real = linalg.norm(w, 2) ** 2

        err: Real = linalg.norm(r)
        if verbose:
            print(
                f"Iteration {iter_ + 1}:\n", x0, f"\nIteration {iter_ + 1}\nerr: {err}"
            )
        if abs(beta) < tol and err < tol:
            return np.asarray(x0)
        s: Real = beta / alpha
        v: ndarray = precondition @ w + s * v
        alpha: Real = beta
    else:
        raise Exception("Maximum iteration exceed.")


def main():
    size = 10
    a = np.random.random((size, size))
    a += np.eye(size) * size
    C = np.diag(np.diag(a) ** -0.5)
    # C = np.eye(size)
    b = (np.random.random(size) - 0.5) * size

    x = conjugate_gradient(a, b, 0, C, verbose=True)
    print(linalg.norm(a @ x - b))


if __name__ == "__main__":
    main()
