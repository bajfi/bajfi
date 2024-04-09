import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def polyfunc(x, z, Q):
    cur, ans = 1, 0
    for i in range(z.shape[0]):
        ans += cur * Q[i]
        cur *= x - z[i]
    return ans


def hermite(x, f_x, f1_x, verbose: bool = False):
    z = np.repeat(x, 2)
    fx = np.repeat(f_x, 2)
    data = np.ones((z.shape[0], z.shape[0] + 1), dtype=np.float_) * np.nan
    data[:, 0] = z
    data[:, 1] = fx
    for i in range(z.shape[0] - 1):
        data[i][2] = (
            ((data[i + 1][1] - data[i][1]) / (data[i + 1][0] - data[i][0]))
            if i & 1
            else f1_x[i // 2]
        )
    for c in range(3, data.shape[1]):
        for r in range(data.shape[0] - c + 1):
            data[r][c] = (data[r + 1][c - 1] - data[r][c - 1]) / (
                data[r + c - 1][0] - data[r][0]
            )
    if verbose:
        df = pd.DataFrame(
            data, columns=["z", "f(z)"] + [f"f_{i}(z)" for i in range(1, data.shape[0])]
        )
        print(df)
    return lambda x: polyfunc(x, z, data[0][1:])


def main():
    x = np.linspace(1, 8, 10)
    # fx = [.620086, .4554022, .2818186]
    # f1_x = [-.522023, -.5698959, -.5811571]
    fx = np.sin(x)
    f1_x = np.cos(x)
    f = hermite(x, fx, f1_x, verbose=True)

    plt.plot(x, fx, "ro", ms=4, label="original points")
    xx = np.linspace(x[0], x[-1], 5 * len(x))
    plt.plot(xx, np.sin(xx), label="exact value")
    plt.plot(xx, f(xx), "--", label="interpolation", alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
