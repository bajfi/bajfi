from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def rk4_linear(f: Sequence[FunctionType], init: Sequence[Real], t_pre: Real, h: Real):
    """
    Kutta, Martin (1901).
    "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen".
    Zeitschrift für Mathematik und Physik. 46: 435–453.
    """
    assert h > 0
    if not isinstance(init, np.ndarray):
        init = np.asarray(init)
    assert len(f) == init.shape[0]
    N = len(f)
    w = np.asarray([h * f[i](t_pre, *init) for i in range(N)])
    w1 = np.asarray([h * f[i](t_pre + h / 2, *(init + 1 / 2 * w)) for i in range(N)])
    w2 = np.asarray(
        [h * f[i](t_pre + 1 / 2 * h, *(init + 1 / 2 * w)) for i in range(N)]
    )
    w3 = np.asarray([h * f[i](t_pre + h, *(init + w)) for i in range(N)])
    yy = init + 1 / 6 * (w + 2 * w1 + 2 * w2 + w3)
    return yy


def ralston4_linear(
    f: Sequence[FunctionType], init: Sequence[Real], t_pre: Real, h: Real
):
    """
    this method has minimum truncate error
    Kutta, Martin (1901).
    "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen".
    Zeitschrift für Mathematik und Physik. 46: 435–453.
    """
    assert h > 0
    if not isinstance(init, np.ndarray):
        init = np.asarray(init)
    assert len(f) == init.shape[0]
    N = len(f)
    w = np.asarray([h * f[i](t_pre, *init) for i in range(N)])
    w1 = np.asarray([h * f[i](t_pre + 0.4 * h, *(init + 0.4 * w)) for i in range(N)])
    w2 = np.asarray(
        [
            h
            * f[i](
                t_pre + 0.45573725 * h,
                *(init + 0.29697761 * w + 0.15875964 * w1),
            )
            for i in range(N)
        ]
    )
    w3 = np.asarray(
        [
            h
            * f[i](
                t_pre + h,
                *(init + 0.21810040 * w - 3.05096516 * w1 + 3.83286476 * w2),
            )
            for i in range(N)
        ]
    )
    yy: ndarray = (
        init + 0.17476028 * w - 0.55148066 * w1 + 1.20553560 * w2 + 0.17118478 * w3
    )
    return yy


def main():
    # f1_0 = lambda t_, y1_, y2_: -4 * y1_ + 3 * y2_ + 6
    # f2_0 = lambda t_, y1_, y2_: -2.4 * y1_ + 1.6 * y2_ + 3.6
    # f1_exact = lambda t_: -3.375 * np.exp(-2 * t_) + 1.875 * np.exp(-0.4 * t_) + 1.5
    # f2_exact = lambda t_: -2.25 * np.exp(-2 * t_) + 2.25 * np.exp(-0.4 * t_)
    # f = [f1_0, f2_0]
    # f_exact = [f1_exact, f2_exact]
    #
    # a, b = 0.0, 10.
    # N = 100
    # h = (b - a) / N
    # init = [0, 0]
    # t = np.linspace(a, b, N + 1)
    #
    # fig, axes = plt.subplots(
    #     2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    # )
    #
    # yy_exact = [f_exact[i](t) for i in range(len(f_exact))]
    # for i in range(len(f_exact)):
    #     axes[0].plot(t, yy_exact[i], "o--", ms=2, label=f"y{i + 1} exact value")
    #
    # for method in [rk4_linear, ralston4_linear]:
    #     yy = [init]
    #     for i in range(N):
    #         yy.append(method(f, yy[-1], t[i], h))
    #
    #     yy = np.asarray(yy).T
    #     for i in range(yy.shape[0]):
    #         axes[0].plot(t, yy[i], "o", ms=2, label=f"{method.__name__} y{i + 1}")
    #         err = np.abs(yy[i] - yy_exact[i])
    #         axes[1].plot(t, err, "o", ms=2, label=f"{method.__name__} y{i + 1} error")
    #
    # axes[0].legend()
    # axes[1].legend()
    # axes[1].set_yscale("log")
    # plt.suptitle("Solving electronic circuit problem\n"
    #              "$2I_1(t)+6[I_1(t)-I_2(t)]+2I_1(t)' =12$\n"
    #              "$\\frac{1}{0.5}\int I_2(t)dt+4I_2(t)+6[I_2(t)-I_1(t)]=0$")
    # plt.show()

    # ==============================================================

    # solving high order different equation
    f1_0 = lambda t_, y1_, y2_, y3_: y2_
    f2_0 = lambda t_, y1_, y2_, y3_: y3_
    f3_0 = (
        lambda t_, y1_, y2_, y3_: 5 * np.log(t_)
        + 9
        + y3_ / t_
        - 3 * y2_ / t_**2
        + 4 * y1_ / t_**3
    )
    f1_exact = (
        lambda t_: -(t_**2)
        + t_ * np.cos(np.log(t_))
        + t_ * np.sin(np.log(t_))
        + t**3 * np.log(t)
    )
    f = [f1_0, f2_0, f3_0]
    f_exact = [f1_exact]

    a, b = 1.0, 5
    N = 40
    h = (b - a) / N
    init = np.array([0.0, 1.0, 3.0])
    t = np.linspace(a, b, N + 1)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )

    yy_exact = [f_exact[i](t) for i in range(len(f_exact))]
    for i in range(len(f_exact)):
        axes[0].plot(t, yy_exact[i], "o--", ms=2, label=f"y{i + 1} exact value")

    for method in [rk4_linear, ralston4_linear]:
        yy = [init]
        for i in range(N):
            yy.append(method(f, yy[-1], t[i], h))

        yy = np.asarray(yy).T
        for i in range(len(yy_exact)):
            axes[0].plot(t, yy[i], "o", ms=2, label=f"{method.__name__} y{i + 1}")
            err = np.abs(yy[i] - yy_exact[i])
            axes[1].plot(t, err, "o", ms=2, label=f"{method.__name__} y{i + 1} error")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.suptitle(
        "Solving high order different equation\n"
        "$t^3{y}'''-t^2{y}''+3ty'-4y=5t^3ln(t)+9t^3$"
    )
    plt.show()


if __name__ == "__main__":
    main()
