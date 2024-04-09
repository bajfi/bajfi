import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Divided_difference:
    def __init__(self, x: np.ndarray, fx: np.ndarray, verbose=False):
        x = np.asarray(x)
        fx = np.asarray(fx)
        self.table = pd.DataFrame(
            {
                "x": x,
                "f[x_i]": fx,
                **{f"f[x_(i-{i})]": np.nan for i in range(1, x.shape[0])},
            }
        )
        for c in range(1, x.shape[0]):
            for r in range(x.shape[0] - c):
                self.table.iat[r, c + 1] = (
                    self.table.iat[r + 1, c] - self.table.iat[r, c]
                ) / (x[r + c] - x[r])

        if verbose:
            print(self.table)

        self.polyfunc_forward = lambda a: self.__polynomia_forward(a, 0, 1)
        self.polyfunc_backward = lambda a: self.__polynomia_backward(a, 0, 1)

    def __expression_backward(self, i, cur):
        if i >= self.table.shape[0]:
            return ""
        return (
            "{:+.6f}".format(self.table.iat[-i - 1, i + 1])
            + cur
            + self.__expression_backward(
                i + 1, cur + f"(x - {self.table.iat[-i - 1, 0]})"
            )
        )

    def __expression_forward(self, i, cur):
        if i >= self.table.shape[0]:
            return ""
        return (
            "{:+.6f}".format(self.table.iat[0, i + 1])
            + cur
            + self.__expression_forward(i + 1, cur + f"(x - {self.table.iat[i, 0]})")
        )

    def __expression_central(self, i, cur):
        N = self.table.shape[0]
        if i >= N:
            return ""
        row = (N - i) // 2
        if 1 & (N - i):
            return (
                "{:+.6f}".format(self.table.iat[row, i + 1])
                + cur
                + self.__expression_central(
                    i + 1, cur + f"(x - {self.table.iat[row, 0]})"
                )
            )

    @property
    def expression_forward(self) -> str:
        return self.__expression_forward(0, "")

    @property
    def expression_backward(self) -> str:
        return self.__expression_backward(0, "")

    @property
    def expression_central(self) -> str:
        return self.__expression_central(0, "")

    def __polynomia_forward(self, xx, i, cur):
        if i >= self.table.shape[0]:
            return 0
        return self.table.iat[0, i + 1] * cur + self.__polynomia_forward(
            xx, i + 1, cur * (xx - self.table.iat[i, 0])
        )

    def __polynomia_backward(self, xx, i, cur):
        if i >= self.table.shape[0]:
            return 0
        return self.table.iat[-i - 1, i + 1] * cur + self.__polynomia_backward(
            xx, i + 1, cur * (xx - self.table.iat[-i - 1, 0])
        )

    def add_point(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        n = self.table.shape[0]
        df = pd.DataFrame(
            {
                "x": x,
                "f[x_i]": y,
                **{f"f[x_(i-{i})]": np.nan for i in range(1, n + x.shape[0])},
            }
        )
        self.table = pd.concat((self.table, df), axis=0, ignore_index=True)
        for r in range(n - 1, self.table.shape[0] - 1):
            for c in range(r + 1):
                self.table.iat[r - c, c + 2] = (
                    self.table.iat[r - c, c + 1] - self.table.iat[r - c + 1, c + 1]
                ) / (self.table.iat[r - c, 0] - self.table.iat[r - c + 1, 0])


def main():
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True, sharex=True
    )
    x = [1.0, 1.3, 1.6]
    # , 1.9, 2.2])
    fx = [0.7651977, 0.6200860, 0.4554022]
    # , 0.2818186, .1103623])
    df = Divided_difference(x, fx, verbose=True)
    axes[0].plot(x, fx, "ro-", ms=3, lw=1)
    xx = np.linspace(0, 3, 30)
    axes[0].plot(xx, df.polyfunc_forward(xx), "g*--", ms=2, lw=1, label="forward")
    axes[0].plot(xx, df.polyfunc_backward(xx), "ro:", ms=2, lw=1, label="backward")
    # axes[0].set_title('\n'.join(wrap(f'$f(x) = {df.expression_forward}$', width=75)))
    axes[0].legend()

    df.add_point([1.9, 2.2], [0.2818186, 0.1103623])
    df.add_point([x[-1]], [fx[-1]])
    axes[1].plot(x + [1.9, 2.2], fx + [0.2818186, 0.1103623], "ro-", ms=3, lw=1)
    axes[1].plot(xx, df.polyfunc_forward(xx), "g*--", ms=2, lw=1, label="forward")
    axes[1].plot(xx, df.polyfunc_backward(xx), "ro:", ms=2, lw=1, label="backward")
    # axes[1].set_title('\n'.join(wrap(f'f(x) = {df.expression_forward}', width=75)))
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
