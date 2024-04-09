import sys

import numpy as np

sys.path.append("../chapter6")
from scipy import sparse
import matplotlib.pyplot as plt

from chapter6.factorization import crout


def interval_func(xx, x, *args):
    """
    :param xx:the interpolation value
    :param x:the original nodes
    :param *args: calculated [a,b,c,d]
    """
    x = np.asarray(x, dtype=np.float_)
    xx = np.asarray(xx)
    ans = np.zeros_like(xx)
    if np.any(xx < x[0]) or np.any(xx > x[-1]):
        raise ValueError("the interpolation value should be within bound")
    for i in range(x.shape[0] - 1):
        pos = (xx >= x[i]) & (xx <= x[i + 1])
        xx_pos = xx[pos]
        ans[pos] = sum(arg[i] * (xx_pos - x[i]) ** j for j, arg in enumerate(args))
    return ans


def cubic_spine_nature(x, y, verbose: bool = False):
    """
    p150
    """
    # f_i = a_i + b_i*(x-x_i) + c_i*(x-x_i)**2 + d_i*(x-x_i)**3
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    A = sparse.diags(
        [
            h[:-1] + [0],
            [1] + [2 * (h[i] + h[i + 1]) for i in range(n - 2)] + [1],
            [0] + h[1:],
        ],
        [-1, 0, 1],
    ).toarray()
    # right hand side of the equation
    B = np.atleast_2d(
        [0]
        + [
            3 / h[i + 1] * (y[i + 2] - y[i + 1]) - 3 / h[i] * (y[i + 1] - y[i])
            for i in range(n - 2)
        ]
        + [0]
    )
    # augment matrix
    A = np.hstack((A, B.T))
    # solve tri-diagonal matrix use crout method
    c = crout(A)
    # calculate b_i and d_i by c
    b = [
        1 / h[i] * (y[i + 1] - y[i]) - h[i] / 3 * (c[i + 1] + 2 * c[i])
        for i in range(n - 1)
    ]
    d = [1 / 3 / h[i] * (c[i + 1] - c[i]) for i in range(n - 1)]
    # print expression if verbose is True
    if verbose:
        for i in range(n - 1):
            print(f"x in [{x[i]},{x[i + 1]}]:")
            print(
                f"S(x) = {y[i]:+.6f}{b[i]:+.6f}(x-{x[i]}){c[i]:+.6f}(x-{x[i]})^2{d[i]:+.6f}(x-{x[i]})^3"
            )
    return lambda a: interval_func(a, x, y, b, c, d), b, c, d


def cubic_spine_clamp(x, y, fp0, fpn, verbose: bool = False):
    """
    p155
    """
    # f_i = a_i + b_i*(x-x_i) + c_i*(x-x_i)**2 + d_i*(x-x_i)**3
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    A = sparse.diags(
        [
            h,
            [2 * h[0]] + [2 * (h[i] + h[i + 1]) for i in range(n - 2)] + [2 * h[-1]],
            h,
        ],
        [-1, 0, 1],
    ).toarray()
    # right hand side of the equation
    B = np.atleast_2d(
        [3 / h[0] * (y[1] - y[0]) - 3 * fp0]
        + [
            3 / h[i + 1] * (y[i + 2] - y[i + 1]) - 3 / h[i] * (y[i + 1] - y[i])
            for i in range(n - 2)
        ]
        + [3 * fpn - 3 / h[-1] * (y[-1] - y[-2])]
    )
    # augment matrix
    A = np.hstack((A, B.T))
    # solve tri-diagonal matrix use crout method
    c = crout(A)
    # calculate b_i and d_i by c
    b = [
        1 / h[i] * (y[i + 1] - y[i]) - h[i] / 3 * (c[i + 1] + 2 * c[i])
        for i in range(n - 1)
    ]
    d = [1 / 3 / h[i] * (c[i + 1] - c[i]) for i in range(n - 1)]
    # print expression if verbose is True
    if verbose:
        for i in range(n - 1):
            print(f"x in [{x[i]},{x[i + 1]}]:")
            print(
                f"S(x) = {y[i]:+.6f}{b[i]:+.6f}(x-{x[i]}){c[i]:+.6f}(x-{x[i]})^2{d[i]:+.6f}(x-{x[i]})^3"
            )
    return lambda a: interval_func(a, x, y, b, c, d), b, c, d


def main():
    # x = np.arange(20)
    # y = np.cos(x)
    x = np.array(
        [
            0.9,
            1.3,
            1.9,
            2.1,
            2.6,
            3.0,
            3.9,
            4.4,
            4.7,
            5,
            6,
            7,
            8,
            9.2,
            10.5,
            11.3,
            11.6,
            12,
            12.6,
            13,
            13.3,
        ]
    )
    y = np.array(
        [
            1.3,
            1.5,
            1.85,
            2.1,
            2.6,
            2.7,
            2.4,
            2.15,
            2.05,
            2.1,
            2.25,
            2.3,
            2.25,
            1.95,
            1.4,
            0.9,
            0.7,
            0.6,
            0.5,
            0.4,
            0.25,
        ]
    )
    f, b, c, d = cubic_spine_nature(x, y, verbose=True)
    # f, b, c, d = cubic_spine_clamp(x, y, 0., -np.sin(19), verbose=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), width_ratios=[2, 3], constrained_layout=True
    )
    xx = np.linspace(x[0], x[-1], x.shape[0] * 10)
    yy = f(xx)
    axes[0].plot(x, y, "ro", ms=4, label="origin nodes")
    axes[0].plot(xx, yy, "d", ms=2, label="interpolation value")
    axes[0].axis("equal")
    axes[0].legend()

    column = ["intervals", "S(x)"]
    cell_text = []
    for i in range(x.shape[0] - 1):
        cell_text.append(
            [
                f"[ {round(x[i], 5)},{round(x[i + 1], 5)} ]",
                "$"
                + f"{y[i]:+.6f}{b[i]:+.6f}(x-{x[i]}){c[i]:+.6f}(x-{x[i]})^2{d[i]:+.6f}(x-{x[i]})^3"
                + "$",
            ]
        )
    axes[1].table(
        cellText=cell_text,
        colLabels=column,
        loc="center",
        cellLoc="center",
        colWidths=[0.3, 0.7],
        edges="closed",
    )
    axes[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
