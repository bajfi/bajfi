from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import numpy as np
from numpy import linalg, ndarray


def deepest_sescent(
    f: Sequence[FunctionType],
    jacobian: FunctionType,
    x0: Sequence[Real],
    tol: Real = 1e-3,
    maxIter: int = 100,
    verbose: bool = False,
):
    """
    p658
    the deepest descent method is not sensitive with the initial
    guess x0, but the convergent rate is only linear
    """
    N: int = len(f)
    assert len(x0) == N
    # g is equal to sum(f_i ** 2)
    g = lambda x0_: sum(f[i](*x0_) ** 2 for i in range(N))
    # the jacobian matrix is need here for gradient calculation
    grad = lambda x0_: np.matmul([f[i](*x0_) for i in range(N)], jacobian(*x0_)) * 2
    x0: ndarray = np.asarray(x0, dtype=np.float_)
    for i in range(maxIter):
        gx0: Real = g(x0)
        # current gradient
        z: ndarray = grad(x0)
        z_norm: Real = linalg.norm(z)
        if z_norm == 0:
            return x0
        z /= z_norm  # normalize
        # choose alpha1 (0), alpha2 and alpha3
        # until g(alpha3) < g(alpha1)
        alpha3: Real = 1.0
        g_alpha3: Real = g(x0 - alpha3 * z)
        while g_alpha3 > gx0:
            alpha3 /= 2
            g_alpha3 = g(x0 - alpha3 * z)
            if alpha3 < tol / 10:
                print("minimum step exceed !!")
                return x0
        # polynomial interpolation with alpha1, alpha2 and alpha3
        # divide difference method is use here to get expression
        g_alpha2: Real = g(x0 - 0.5 * alpha3 * z)  # g_alpha1 = gx0, alpha2 = 0.5*alpha3
        h1: Real = 2 * (g_alpha2 - gx0) / alpha3
        h2: Real = 2 * (g_alpha3 - g_alpha2) / alpha3
        h3: Real = (h2 - h1) / alpha3
        # the expression is g_alpha1 + h1(x-alpha2) + h3*(x-0)*(x-alpha2)
        # after simplify, the critical point is (alpha3*h3 - 2h1)/(4*h3)
        alpha: Real = 0.25 * (alpha3 * h3 - 2 * h1) / h3
        # step forward
        x1: Real = x0 - alpha * z
        err: Real = linalg.norm(x1 - x0)
        if verbose:
            print(
                f"Iteration {i + 1}\nx:\n",
                x0,
                "\n||x1 - x0||:\n",
                err,
                "\ng(x):\n",
                g(x1),
            )
        if err < tol:
            return x1
        x0 = x1
    else:
        raise Exception("maximum iteration exceeded !!")


def main():
    # f1 = lambda x1_, x2_, x3_: 3 * x1_ - np.cos(x2_ * x3_) - 0.5
    # f2 = lambda x1_, x2_, x3_: x1_ ** 2 - 81 * (x2_ + 0.1) ** 2 + np.sin(x3_) + 1.06
    # f3 = lambda x1_, x2_, x3_: np.exp(-x1_ * x2_) + 20 * x3_ + (10 * np.pi - 3) / 3
    #
    # jacobian = lambda x1_, x2_, x3_: [
    #     [3.0, x3_ * np.sin(x2_ * x3_), x2_ * np.sin(x2_ * x3_)],
    #     [2 * x1_, -162 * (x2_ + 0.1), np.cos(x3_)],
    #     [-x2_ * np.exp(-x1_ * x2_), -x1_ * np.exp(-x1_ * x2_), 20],
    # ]

    f1 = lambda x1_, x2_, x3_: 2 * x1_ - 3 * x2_ + x3_ - 4
    f2 = lambda x1_, x2_, x3_: 2 * x1_ + x2_ - x3_ + 4
    f3 = lambda x1_, x2_, x3_: x1_**2 + x2_**2 + x3_**2 - 4

    jacobian = lambda x1_, x2_, x3_: [
        [2.0, -3, 1],
        [2, 1, -1],
        [2 * x1_, 2 * x2_, 2 * x3_],
    ]

    deepest_sescent([f1, f2, f3], jacobian, [0.0, 0, 0], verbose=True)


if __name__ == "__main__":
    main()
