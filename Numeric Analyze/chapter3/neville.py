import numpy as np
import pandas as pd


def neville(xi: np.ndarray, fx: np.ndarray, x):
    xi = np.asarray(xi)
    fx = np.asarray(fx)
    df = pd.DataFrame(
        {
            "x_i": xi,
            "x-x_i": x - xi,
            "Q_i0": fx,
            **{f"Q_x{i}": np.nan for i in range(1, xi.shape[0])},
        }
    )
    k = df.shape[-1] - xi.shape[0]
    for i in range(1, xi.shape[0]):
        for j in range(1, i + 1):
            df.iat[i, j + k] = (
                (x - xi[i - j]) * df.iat[i, j - 1 + k]
                - (x - xi[i]) * df.iat[i - 1, j - 1 + k]
            ) / (xi[i] - xi[i - j])
    return df


def main():
    # assume the function is exp(x)
    N = 5
    x = np.arange(5)
    fx = np.exp(x)
    df = neville(x, fx, 2.1)
    print(df)
    print(np.exp(2.1))


if __name__ == "__main__":
    main()
