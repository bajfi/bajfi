from numbers import Real
from types import FunctionType

import numpy as np


def composite_simpson(f: FunctionType, a: Real, b: Real, n: int):
    assert n > 1 and n % 2 == 0
    a, b = min(a, b), max(a, b)
    x = np.linspace(a, b, n + 1)
    mask = np.arange(n + 1) % 2 == 0
    X_even = np.sum(f(x[mask]))
    X_odd = np.sum(f(x[~mask]))
    return (b - a) / n / 3 * (2 * X_even + 4 * X_odd - f(a) - f(b))


def composite_trapezoidal(f: FunctionType, a: Real, b: Real, n: int):
    assert n > 0
    a, b = min(a, b), max(a, b)
    x = np.linspace(a, b, n + 1)
    return (b - a) / n / 2 * (2 * f(x).sum() - f(a) - f(b))


def main():
    f = lambda x: np.sin(x) * x
    a, b = 0, 2 * np.pi
    ans_s = composite_simpson(f, a, b, 20)
    ans_t = composite_trapezoidal(f, a, b, 1000)
    print(ans_s)
    print(ans_t)


if __name__ == "__main__":
    main()
