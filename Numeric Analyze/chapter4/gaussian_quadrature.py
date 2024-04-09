from numbers import Real
from types import FunctionType

import numpy as np


def roots():
    return [
        np.asarray([0]),
        np.asarray([np.sqrt(3) / 3, -np.sqrt(3) / 3]),
        np.asarray([np.sqrt(3 / 5), 0.0, -np.sqrt(3 / 5)]),
        np.asarray(
            [
                np.sqrt(3 / 7 + (2 / 7) * (6 / 5)),
                np.sqrt(3 / 7 - (2 / 7) * (6 / 5)),
                -np.sqrt(3 / 7 - (2 / 7) * (6 / 5)),
                -np.sqrt(3 / 7 + (2 / 7) * (6 / 5)),
            ]
        ),
        np.asarray(
            [
                1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
                1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                0,
                -1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                -1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
            ]
        ),
    ]


def coefs():
    return [
        np.asarray([2]),
        np.asarray([1.0, 1.0]),
        np.asarray([5 / 9, 8 / 9, 5 / 9]),
        np.asarray(
            [
                (18 - np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 - np.sqrt(30)) / 36,
            ]
        ),
        np.asarray(
            [
                (322 - 13 * np.sqrt(70)) / 900,
                (322 + 13 * np.sqrt(70)) / 900,
                128 / 225,
                (322 + 13 * np.sqrt(70)) / 900,
                (322 - 13 * np.sqrt(70)) / 900,
            ]
        ),
    ]


def gaussian_quadrature(f: FunctionType, x1: Real, x2: Real, n: int):
    assert n > 0
    x1, x2 = min(x1, x2), max(x1, x2)
    # convert f to g
    g = lambda t: f(((x2 - x1) * t + x1 + x2) / 2)
    root, coef = roots(), coefs()
    return (g(root[n - 1]) * coef[n - 1]).sum() * (x2 - x1) / 2


def main():
    f = lambda x: x**2
    a, b = 0, 3
    print(gaussian_quadrature(f, a, b, 5))


if __name__ == "__main__":
    main()
