from types import FunctionType

import numpy as np


def bisection(f: FunctionType, low, high, tol, maxIter: int = 20, verbose=False):
    if np.sign(f(low)) == np.sign(f(high)):
        raise Exception("f(low) and f(high) should be sign different")
    if tol < 0:
        raise Exception("tol should be larger than 0")
    if low > high:
        low, high = high, low
    if f(low) == 0:
        return low
    if f(high) == 0:
        return high
    p0 = low
    for i in range(maxIter):
        p = low + (high - low) / 2
        if verbose:
            print(f"Iteration {i + 1}: {p}")
        if f(p) == 0 or np.allclose(p, p0, rtol=tol):
            return p
        if np.sign(f(low)) == np.sign(f(p)):
            low = p
        else:
            high = p
        p0 = p
    else:
        raise Exception("maximum iteration exceed")


def main():
    def f(x):
        return (x + 1) * (x - 0.5) * (x - 1)

    p = bisection(f, low=-1.25, high=3.25, tol=1e-5, maxIter=50, verbose=True)
    print(p, f(p))


if __name__ == "__main__":
    main()
