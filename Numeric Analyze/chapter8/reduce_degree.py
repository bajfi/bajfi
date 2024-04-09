import numpy as np
import matplotlib.pyplot as plt


# take f(x) = xe^x for example
# approximate it with degree 6 by Taylor polynomial


def main():
    f_exact = lambda x: x * np.exp(x)
    f_taylor = (
        lambda x: x + x**2 + x**3 / 2 + x**4 / 6 + x**5 / 24 + x**6 / 120
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True, sharex="col"
    )
    xx = np.linspace(0, 2)

    yy_exact = f_exact(xx)
    axes[0].plot(xx, yy_exact, lw=1, label="exact value")

    yy_taylor = f_taylor(xx)
    axes[0].plot(xx, yy_taylor, "o", ms=2, label="Taylor expansion")
    err_taylor = np.abs(yy_taylor - yy_exact)
    axes[1].plot(xx, err_taylor, "o--", ms=2, label="Taylor expansion - 6")

    # we use chebyshev polynomial to reduce the degree
    f_che5 = (
        lambda x: f_taylor(x) - (32 * x**6 - 48 * x**4 + 18 * x**2 - 1) / 32 / 120
    )
    yy_che5 = f_che5(xx)
    axes[0].plot(xx, yy_che5, "o", ms=2, label="chebyshev - 5")
    err_che5 = np.abs(yy_che5 - yy_exact)
    axes[1].plot(xx, err_che5, "o--", ms=2, label="chebyshev - 5")

    f_che4 = lambda x: f_che5(x) - (16 * x**5 - 20 * x**3 + 5 * x) / 16 / 24
    yy_che4 = f_che4(xx)
    axes[0].plot(xx, yy_che4, "o", ms=2, label="chebyshev - 4")
    err_che4 = np.abs(yy_che4 - yy_exact)
    axes[1].plot(xx, err_che4, "o--", ms=2, label="chebyshev - 4")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
