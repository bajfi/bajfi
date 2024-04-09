from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np


def mid_point(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    the midPoint method is Runge-Kutta of order 2
    """
    assert h > 0
    yy = init + h * f(t_pre + h / 2, init + h / 2 * f(t_pre, init))
    return yy


def modified_euler(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    the modified Euler method is another form of runge-kutta of order 2
    """
    assert h > 0
    yy = init + h / 2 * (f(t_pre, init) + f(t_pre + h, init + h * f(t_pre, init)))
    return yy


def rk3_generic(f: FunctionType, init: Real, t_pre: Real, h: Real, alpha: Real = 0.5):
    """
    Sanderse, Benjamin; Veldman, Arthur (2019).
     "Constraint-consistent Runge–Kutta methods
     for one-dimensional incompressible multiphase flow".
      J. Comput. Phys. 384: 170. arXiv:1809.06114. Bibcode:2019JCoPh.384.170S
      doi:10.1016/j.jcp.2019.02.001. S2CID 73590909
    """
    assert h > 0
    assert alpha not in (0, 2 / 3, 1)
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + alpha * h, init + alpha * w)
    w2 = h * f(
        t_pre + h,
        init
        + (1 + (1 - alpha) / alpha / (3 * alpha - 2)) * w
        - (1 - alpha) / alpha / (3 * alpha - 2) * w1,
    )
    yy = (
        init
        + (0.5 - 1 / 6 / alpha) * w
        + 1 / 6 / alpha / (1 - alpha) * w1
        + (2 - 3 * alpha) / 6 / (1 - alpha) * w2
    )
    return yy


def heun(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    the heun method is another form of runge-kutta of order 3
    """
    assert h > 0
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + h / 3, init + 1 / 3 * w)
    w2 = h * f(t_pre + 2 / 3 * h, init + 2 / 3 * w1)
    yy = init + 1 / 4 * (w + 3 * w2)
    return yy


def ssprk3(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    Third-order Strong Stability Preserving Runge-Kutta
    """
    assert h > 0
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + h, init + w)
    w2 = h * f(t_pre + 1 / 2 * h, init + 1 / 4 * (w + w1))
    yy = init + 1 / 6 * (w + w1 + 4 * w2)
    return yy


def ralston_3(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    Ralston, Anthony (1962).
    "Runge-Kutta Methods with Minimum Error Bounds".
    Math. Comput. 16 (80): 431–437.
    doi:10.1090/S0025-5718-1962-0150954-0.
    """
    assert h > 0
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + 1 / 2 * h, init + 1 / 2 * w)
    w2 = h * f(t_pre + 3 / 4 * h, init + 3 / 4 * w1)
    yy = init + 1 / 9 * (2 * w + 3 * w1 + 4 * w2)
    return yy


def rk4(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    Kutta, Martin (1901).
    "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen".
    Zeitschrift für Mathematik und Physik. 46: 435–453.
    """
    assert h > 0
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + h / 2, init + 1 / 2 * w)
    w2 = h * f(t_pre + 1 / 2 * h, init + 1 / 2 * w1)
    w3 = h * f(t_pre + h, init + w2)
    yy = init + 1 / 6 * (w + 2 * w1 + 2 * w2 + w3)
    return yy


def ralston_4(f: FunctionType, init: Real, t_pre: Real, h: Real):
    """
    this method has minimum truncate error
    Kutta, Martin (1901).
    "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen".
    Zeitschrift für Mathematik und Physik. 46: 435–453.
    """
    assert h > 0
    w = h * f(t_pre, init)
    w1 = h * f(t_pre + 0.4 * h, init + 0.4 * w)
    w2 = h * f(t_pre + 0.45573725 * h, init + 0.29697761 * w + 0.15875964 * w1)
    w3 = h * f(t_pre + h, init + 0.21810040 * w - 3.05096516 * w1 + 3.83286476 * w2)
    yy = init + 0.17476028 * w - 0.55148066 * w1 + 1.20553560 * w2 + 0.17118478 * w3
    return yy


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0, 2
    N = 20
    h = (b - a) / N
    init = 0.5
    t = np.linspace(a, b, N + 1)
    yy_exact = f_exact(t)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    axes[0].plot(t, f_exact(t), ":", label="exact value")
    for method in [mid_point, modified_euler, heun, ssprk3, ralston_3, rk4, ralston_4]:
        yy = [init]
        for i in range(N):
            yy.append(method(f_0, yy[-1], t[i], h))
        axes[0].plot(t, yy, "o", ms=2, label=method.__name__)
        err = np.abs(np.asarray(yy) - yy_exact)
        axes[1].plot(t, err, "o--", ms=1.5, label=method.__name__)

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
