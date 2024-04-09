from numbers import Real
from types import FunctionType

import numpy as np

from composite import composite_simpson, composite_trapezoidal


def adaptive_quadrature(
    f: FunctionType,
    a: Real,
    b: Real,
    method: FunctionType = composite_simpson,
    level: int = 10,
    tol=1e-5,
):
    """
    p225
    """
    if level <= 0:
        raise Exception("level exceeded")
    S = method(f, a, b, 2)
    S1 = method(f, a, (a + b) / 2, 2)
    S2 = method(f, (a + b) / 2, b, 2)
    if abs(S - S1 - S2) < tol:
        return S1 + S2
    return adaptive_quadrature(
        f, a, (a + b) / 2, method, level - 1, tol
    ) + adaptive_quadrature(f, (a + b) / 2, b, method, level - 1, tol)


def main():
    f = lambda x: 100 / x**2 * np.sin(10 / x)
    a, b = 1.0, 3.0
    print(adaptive_quadrature(f, a, b, composite_trapezoidal, 10))
    print(adaptive_quadrature(f, a, b, composite_simpson, 10))


if __name__ == "__main__":
    main()
