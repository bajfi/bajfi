from types import FunctionType

import numpy as np
import pandas as pd

from composite import composite_trapezoidal


def rombery(f: FunctionType, a, b, n: int, verbose: bool = False):
    """
    p217
    """
    assert isinstance(n, int) and n > 0
    a, b = min(a, b), max(a, b)
    data = np.ones((n, n), dtype=np.float_) * np.nan
    # fill by row
    for r in range(n):
        data[r][0] = composite_trapezoidal(f, a, b, 2**r)
        for c in range(1, r + 1):
            data[r][c] = data[r][c - 1] + (data[r][c - 1] - data[r - 1][c - 1]) / (
                4**c - 1
            )
    if verbose:
        df = pd.DataFrame(data, columns=[f"O(h^{2 * i})" for i in range(1, n + 1)])
        print(df)
    return data[-1][-1]


def main():
    f = lambda x: x * np.sin(x)
    a, b = 0, 2 * np.pi
    rombery(f, a, b, 6, verbose=True)


if __name__ == "__main__":
    main()
