import os

import matplotlib.pyplot as plt
import numpy as np

os.path.join("../chapter3/divided_difference.py")
from chapter3.divided_difference import Divided_difference


def main():
    a, b = 0.0, 3.0
    xx = np.linspace(a, b)
    f = lambda x: x * np.exp(np.cos(x))
    yy = f(xx)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True, sharex="col"
    )
    axes[0].plot(xx, yy, "o", ms=2, label="exact value")

    # use equal gap points to interpolation
    xx_equal = np.linspace(a, b, 10)
    yy_equal = f(xx_equal)
    axes[0].plot(xx_equal, yy_equal, "o", label="equally distribute")
    Df_equal = Divided_difference(xx_equal, yy_equal)
    interp_equal = Df_equal.polyfunc_forward(xx)
    axes[0].plot(xx, interp_equal, "--", label="interpolation equally")
    axes[1].plot(
        xx,
        np.clip(np.abs(interp_equal - yy), 1e-2, 10),
        "o--",
        ms=2,
        label="interpolation equally",
    )

    # use Chebyshev interpolation gap
    xx_chebyshev = 0.5 * (
        (b - a)
        * np.cos(
            0.5 * (2 * np.arange(xx_equal.shape[0]) + 1) * np.pi / xx_equal.shape[0]
        )
        + a
        + b
    )
    yy_chebyshev = f(xx_chebyshev)
    Df_chebyshev = Divided_difference(xx_chebyshev, yy_chebyshev)
    axes[0].plot(xx_chebyshev, yy_chebyshev, "o", label="chebyshev distribute")
    interp_chebyshev = Df_chebyshev.polyfunc_forward(xx)
    axes[0].plot(xx, interp_chebyshev, "--", label="interpolation chebyshev")
    axes[1].plot(xx, np.abs(interp_chebyshev - yy), "o--", ms=2, label="chebyshev step")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
