from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy import ndarray
from scipy import linalg


def power_method(
    A: Sequence[Sequence[Real]],
    x0: Sequence[Real],
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = True,
):
    """
    p580
    """
    A: ndarray = np.asarray(A, dtype=np.float_)
    x0: ndarray = np.asarray(x0)
    assert np.linalg.norm(x0, 1) != 0
    u0, u1 = 0.0, x0[np.argmax(np.abs(x0))]
    x0 /= u1
    for i in range(maxIter):
        y = A @ x0
        v = y[np.argmax(np.abs(y))]
        if v == 0:
            print("A has a eigen value 0, select a new x to restart")
            return 0, x0
        y /= v
        # use Aitken's method to accelerate converge rate
        dominant = v - 2 * u1 + u0
        u_converge = u0 - (u0 - u1) ** 2 / (dominant + np.finfo(dominant).eps)
        if verbose:
            print(
                f"Iteration {i + 1}:\n",
                f"eigen vector = {y}\n",
                f"eigen value = {u_converge}",
            )
        if np.linalg.norm(x0 - y) < tol:
            return u_converge, y
        u0, u1 = u1, v
        x0 = y
    else:
        raise Exception("Maximum iteration exceeded !!")


def symmetry_power_method(
    A: Sequence[Sequence[Real]],
    x0: Sequence[Real],
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = True,
):
    """
    p581
    """
    A: ndarray = np.asarray(A, dtype=np.float_)
    # assert linalg.issymmetric(A)
    x0: ndarray = np.asarray(x0)
    assert np.linalg.norm(x0, 1) != 0
    u0, u1 = 0.0, np.linalg.norm(x0)
    x0 /= u1
    for i in range(maxIter):
        y: ndarray = A @ x0
        if np.linalg.norm(y, 1) == 0:
            print("A has a eigen value 0, select a new x to restart")
            return 0, x0
        v = np.dot(x0, y)
        y /= np.linalg.norm(y)
        # use Aitken's method to accelerate converge rate
        dominant = v - 2 * u1 + u0
        u_converge = u0 - (u0 - u1) ** 2 / (dominant + np.finfo(dominant).eps)
        if verbose:
            print(
                f"Iteration {i + 1}:\n",
                f"eigen vector = {y}\n",
                f"eigen value = {u_converge}",
            )
        if np.linalg.norm(x0 - y) < tol:
            return v, y
        u0, u1 = u1, v
        x0 = y
    else:
        raise Exception("Maximum iteration exceeded !!")


def inverse_power_method(
    A: Sequence[Sequence[Real]],
    x0: Sequence[Real],
    tol: Real = 1e-5,
    maxIter: int = 50,
    verbose: bool = True,
):
    """
    p585
    if A is symmetry, the converge rate may be slow
    """
    A: ndarray = np.asarray(A, dtype=np.float_)
    x0: ndarray = np.asarray(x0)
    assert np.linalg.norm(x0, 1) != 0
    q = np.dot(x0 @ A, x0) / np.dot(x0, x0)
    mat_solve = A - q * np.eye(A.shape[0])
    u0, u1 = 0.0, x0[np.argmax(np.abs(x0))]
    x0 /= u1
    for i in range(maxIter):
        # solving linear equations (A - qI)y = x0
        # LU decompose can be used which complemented in chapter6
        try:
            y = linalg.solve(mat_solve, x0)
        except Exception as e:
            match e:
                case linalg.LinAlgError:
                    print(f"{q} is a eigen value")
                    return q, None
                case _:
                    raise e
        v = y[np.argmax(np.abs(y))]
        y /= v
        # use Aitken's method to accelerate converge rate
        v = 1 / v + q
        dominant = v - 2 * u1 + u0
        u_converge = u0 - (u0 - u1) ** 2 / (dominant + np.finfo(dominant).eps)
        if verbose:
            print(
                f"Iteration {i + 1}:\n",
                f"eigen vector = {y}\n",
                f"eigen value = {u_converge}",
            )
        if np.linalg.norm(x0 - y) < tol:
            return u_converge, y
        u0, u1 = u1, v
        x0 = y
    else:
        raise Exception("Maximum iteration exceeded !!")


def main():
    N = 10
    A = np.array(
        [
            [5.0, -2, -0.5, 1.5],
            [-2, 5, 1.5, -0.5],
            [-0.5, 1.5, 5, -2],
            [1.5, -0.5, -2, 4],
        ]
    )
    x = np.ones(A.shape[0], dtype=np.float_)
    power_method(A, x, verbose=True, maxIter=50)
    # inverse_power_method(A, x, verbose=True, maxIter=50)
    # symmetry_power_method(A, x, verbose=True, maxIter=50)


if __name__ == "__main__":
    main()
