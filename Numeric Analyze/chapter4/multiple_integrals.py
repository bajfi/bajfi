from numbers import Real
from types import FunctionType

import numpy as np

from composite import composite_simpson, composite_trapezoidal
from gaussian_quadrature import coefs, gaussian_quadrature, roots


def double_trapezoid(f: FunctionType, x1: Real, x2: Real, y1, y2, row: int, col: int):
    assert col > 0
    if isinstance(y1, Real):
        return double_trapezoid(f, x1, x2, lambda y: y1, y2, row, col)
    if isinstance(y2, Real):
        return double_trapezoid(f, x1, x2, y1, lambda y: y2, row, col)
    x1, x2 = min(x1, x2), max(x1, x2)
    xx = np.linspace(x1, x2, col + 1)
    return (
        (
            composite_trapezoidal(lambda y: f(x1, y), y1(x1), y2(x1), row)
            + composite_trapezoidal(lambda y: f(x2, y), y1(x2), y2(x2), row)
            + 2
            * sum(
                composite_trapezoidal(lambda y: f(xx[i], y), y1(xx[i]), y2(xx[i]), row)
                for i in range(1, col)
            )
        )
        * (x2 - x1)
        / col
        / 2
    )


def double_simpson(f: FunctionType, x1: Real, x2: Real, y1, y2, row: int, col: int):
    assert col > 0
    if isinstance(y1, Real):
        return double_simpson(f, x1, x2, lambda y: y1, y2, row, col)
    if isinstance(y2, Real):
        return double_simpson(f, x1, x2, y1, lambda y: y2, row, col)
    x1, x2 = min(x1, x2), max(x1, x2)
    xx = np.linspace(x1, x2, col + 1)
    return (
        (
            composite_simpson(lambda y: f(x1, y), y1(x1), y2(x1), row)
            + composite_simpson(lambda y: f(x2, y), y1(x2), y2(x2), row)
            + 4
            * sum(
                composite_simpson(lambda y: f(xx[i], y), y1(xx[i]), y2(xx[i]), row)
                for i in range(1, col, 2)
            )
            + 2
            * sum(
                composite_simpson(lambda y: f(xx[i], y), y1(xx[i]), y2(xx[i]), row)
                for i in range(2, col, 2)
            )
        )
        * (x2 - x1)
        / col
        / 3
    )


def double_gaussian(f: FunctionType, x1: Real, x2: Real, y1, y2, n: int):
    assert n > 0
    if isinstance(y1, Real):
        return double_gaussian(f, x1, x2, lambda y: y1, y2, n)
    if isinstance(y2, Real):
        return double_gaussian(f, x1, x2, y1, lambda y: y2, n)
    root, coef = roots()[n - 1], coefs()[n - 1]
    return gaussian_quadrature(
        lambda x: (y2(x) - y1(x))
        / 2
        * sum(
            coef[i] * f(x, ((y2(x) - y1(x)) * root[i] + y1(x) + y2(x)) / 2)
            for i in range(n)
        ),
        x1,
        x2,
        n,
    )


def main():
    f = lambda x, y: np.log(x + 2 * y)
    x1, x2 = 1.4, 2.0
    y1, y2 = 1.0, 1.5
    print("rectangular region: ")
    print("double_trapezoid: ", double_trapezoid(f, x1, x2, y1, y2, 2, 4))
    print("double_simpson: ", double_simpson(f, x1, x2, y1, y2, 2, 4))
    print("double_gaussian: ", double_gaussian(f, x1, x2, y1, y2, 3))

    a, b = 0.1, 0.5
    c = lambda x: x**3
    d = lambda x: x**2
    f = lambda x, y: np.exp(y / x)
    print("non rectangular region: ")
    print("double_trapezoid: ", double_trapezoid(f, a, b, c, d, 20, 20))
    print("double_simpson: ", double_simpson(f, a, b, c, d, 20, 20))
    print("double_gaussian: ", double_gaussian(f, a, b, c, d, 5))


if __name__ == "__main__":
    main()
