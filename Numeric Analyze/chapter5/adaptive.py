from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np


def bogacki_shampine(
    f: FunctionType,
    a: Real,
    b: Real,
    init: Real,
    hmin: Real,
    hmax: Real,
    Tol: float = 1e-5,
):
    """
    Bogacki, Przemysław; Shampine, Lawrence F. (1989),
     "A 3(2) pair of Runge–Kutta formulas",
      Applied Mathematics Letters, 2 (4): 321–325,
       doi:10.1016/0893-9659(89)90079-7,ISSN 0893-9659
    """
    a, b = min(a, b), max(a, b)
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    assert hmax < b - a
    # generate mesh in N step
    h0 = hmax
    h = [h0]
    t = [a]
    yy = [init]
    w = lambda h_: h_ * f(t[-1], yy[-1])
    w1 = lambda h_: h_ * f(t[-1] + h_ / 2, yy[-1] + 1 / 2 * w(h_))
    w2 = lambda h_: h_ * f(t[-1] + 3 / 4 * h_, yy[-1] + 3 / 4 * w1(h_))
    w3 = lambda h_: h_ * f(
        t[-1] + h_,
        yy[-1] + 2 / 9 * w(h_) + 1 / 3 * w1(h_) + 4 / 9 * w2(h_),
    )
    R = (
        lambda h_: np.abs(
            -5 / 72 * w(h_) + 1 / 12 * w1(h_) + 1 / 9 * w2(h_) - 1 / 8 * w3(h_)
        )
        / h_
    )
    while t[-1] < b:
        # update step size until error less than tol
        r = R(h0)
        while r > Tol:
            # the step used to avoid significant step size changing
            sigma = np.clip((Tol / r / 2) ** 0.5, 0.1, 4)
            h0 = min(hmax, sigma * h0)
            # terminate program when step size reach the minimum size
            if h0 < hmin:
                raise Exception("minimum h exceeded !!")
            r = R(h0)
        else:
            yy.append(
                yy[-1]
                + 7 / 24 * w(h0)
                + 1 / 4 * w1(h0)
                + 1 / 3 * w2(h0)
                + 1 / 8 * w3(h0)
            )
            # step forward
            t.append(t[-1] + h0)
            h.append(h0)
            # initialize h with new suze
            h0 = b - t[-1]
    return np.asarray(t), np.asarray(yy), np.asarray(h)


def rk_fehlberg(
    f: FunctionType,
    a: Real,
    b: Real,
    init: Real,
    hmin: Real,
    hmax: Real,
    Tol: float = 1e-5,
):
    """
    implement of Runge–Kutta–Fehlberg method
    which has order 5 and 4
    """
    a, b = min(a, b), max(a, b)
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    assert hmax < b - a
    # generate mesh in N step
    h0 = hmax
    h = [h0]
    t = [a]
    yy = [init]
    w = lambda h_: h_ * f(t[-1], yy[-1])
    w1 = lambda h_: h_ * f(t[-1] + h_ / 4, yy[-1] + 1 / 4 * w(h_))
    w2 = lambda h_: h_ * f(
        t[-1] + 3 / 8 * h_, yy[-1] + 3 / 32 * w(h_) + 9 / 32 * w1(h_)
    )
    w3 = lambda h_: h_ * f(
        t[-1] + 12 / 13 * h_,
        yy[-1] + 1932 / 2197 * w(h_) - 7200 / 2197 * w1(h_) + 7296 / 2197 * w2(h_),
    )
    w4 = lambda h_: h_ * f(
        t[-1] + h_,
        yy[-1]
        + 439 / 216 * w(h_)
        - 8 * w1(h_)
        + 3680 / 513 * w2(h_)
        - 845 / 4104 * w3(h_),
    )
    w5 = lambda h_: h_ * f(
        t[-1] + 1 / 2 * h_,
        yy[-1]
        - 8 / 27 * w(h_)
        + 2 * w1(h_)
        - 3544 / 2565 * w2(h_)
        + 1859 / 4104 * w3(h_)
        - 11 / 40 * w4(h_),
    )
    R = (
        lambda h_: np.abs(
            w(h_) / 360
            - 128 / 4275 * w2(h_)
            - 2197 / 75240 * w3(h_)
            + 1
            / 50
            * w4(
                h_,
            )
            + 2 / 55 * w5(h_)
        )
        / h_
    )
    while t[-1] < b:
        # update step size until error less than tol
        r = R(h0)
        while r > Tol:
            # the step used to avoid significant step size changing
            sigma = np.clip((Tol / r / 2) ** 0.25, 0.1, 4)
            h0 = min(hmax, sigma * h0)
            # terminate program when step size reach the minimum size
            if h0 < hmin:
                raise Exception("minimum h exceeded !!")
            r = R(h0)
        else:
            yy.append(
                yy[-1]
                + 25 / 216 * w(h0)
                + 1408 / 2565 * w2(h0)
                + 2197 / 4104 * w3(h0)
                - 1 / 5 * w4(h0)
            )
            # step forward
            t.append(t[-1] + h0)
            h.append(h0)
            # initialize h with new suze
            h0 = b - t[-1]
    return np.asarray(t), np.asarray(yy), np.asarray(h)


def cash_karp(
    f: FunctionType,
    a: Real,
    b: Real,
    init: Real,
    hmin: Real,
    hmax: Real,
    Tol: float = 1e-5,
):
    """
    implement of Cash–Karp method
    which has order 5 and 4
    """
    a, b = min(a, b), max(a, b)
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    assert hmax < b - a
    # generate mesh in N step
    h0 = hmax
    h = [h0]
    t = [a]
    yy = [init]
    w = lambda h_: h_ * f(t[-1], yy[-1])
    w1 = lambda h_: h_ * f(t[-1] + h_ / 5, yy[-1] + 1 / 5 * w(h_))
    w2 = lambda h_: h_ * f(
        t[-1] + 3 / 10 * h_, yy[-1] + 3 / 40 * w(h_) + 9 / 40 * w1(h_)
    )
    w3 = lambda h_: h_ * f(
        t[-1] + 3 / 5 * h_, yy[-1] + 3 / 10 * w(h_) - 9 / 10 * w1(h_) + 6 / 5 * w2(h_)
    )
    w4 = lambda h_: h_ * f(
        t[-1] + h_,
        yy[-1] - 11 / 54 * w(h_) + 5 / 2 * w1(h_) - 70 / 27 * w2(h_) + 35 / 27 * w3(h_),
    )
    w5 = lambda h_: h_ * f(
        t[-1] + 7 / 8 * h_,
        yy[-1]
        + 1631 / 55296 * w(h_)
        + 175 / 512 * w1(h_)
        + 575 / 13824 * w2(h_)
        + 44275 / 110592 * w3(h_)
        + 253 / 4096 * w4(h_),
    )
    R = (
        lambda h_: np.abs(
            -277 / 64512 * w(h_)
            + 6925 / 370944 * w2(h_)
            - 6925 / 202752 * w3(h_)
            - 277
            / 14336
            * w4(
                h_,
            )
            + 277 / 7084 * w5(h_)
        )
        / h_
    )
    while t[-1] < b:
        # update step size until error less than tol
        r = R(h0)
        while r > Tol:
            # the step used to avoid significant step size changing
            sigma = np.clip((Tol / r / 2) ** 0.25, 0.1, 4)
            h0 = min(hmax, sigma * h0)
            # terminate program when step size reach the minimum size
            if h0 < hmin:
                raise Exception("minimum h exceeded !!")
            r = R(h0)
        else:
            yy.append(
                yy[-1]
                + 2825 / 27648 * w(h0)
                + 18575 / 48384 * w2(h0)
                + 13525 / 55296 * w3(h0)
                + 277 / 14336 * w4(h0)
                + 1 / 4 * w5(h0)
            )
            # step forward
            t.append(t[-1] + h0)
            h.append(h0)
            # initialize h with new suze
            h0 = b - t[-1]
    return np.asarray(t), np.asarray(yy), np.asarray(h)


def dormand_prince(
    f: FunctionType,
    a: Real,
    b: Real,
    init: Real,
    hmin: Real,
    hmax: Real,
    Tol: float = 1e-5,
):
    """
    implement of Dormand–Prince method
    which has order 5 and 4
    """
    a, b = min(a, b), max(a, b)
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    assert hmax < b - a
    # generate mesh in N step
    h0 = hmax
    h = [h0]
    t = [a]
    yy = [init]
    w = lambda h_: h_ * f(t[-1], yy[-1])
    w1 = lambda h_: h_ * f(t[-1] + h_ / 5, yy[-1] + 1 / 5 * w(h_))
    w2 = lambda h_: h_ * f(
        t[-1] + 3 / 10 * h_, yy[-1] + 3 / 40 * w(h_) + 9 / 40 * w1(h_)
    )
    w3 = lambda h_: h_ * f(
        t[-1] + 4 / 5 * h_,
        yy[-1] + 44 / 45 * w(h_) - 56 / 15 * w1(h_) + 32 / 9 * w2(h_),
    )
    w4 = lambda h_: h_ * f(
        t[-1] + 8 / 9 * h_,
        yy[-1]
        + 19372 / 6561 * w(h_)
        - 25360 / 2187 * w1(h_)
        + 64448 / 6561 * w2(h_)
        - 212 / 729 * w3(h_),
    )
    w5 = lambda h_: h_ * f(
        t[-1] + h_,
        yy[-1]
        + 9017 / 3168 * w(h_)
        - 355 / 33 * w1(h_)
        + 46732 / 5247 * w2(h_)
        + 49 / 176 * w3(h_)
        - 5103 / 18656 * w4(h_),
    )
    w6 = lambda h_: h_ * f(
        t[-1] + h_,
        yy[-1]
        + 35 / 384 * w(h_)
        + 500 / 1113 * w2(h_)
        + 125 / 192 * w3(h_)
        - 2187 / 6784 * w4(h_)
        + 11 / 84 * w5(h_),
    )
    R = (
        lambda h_: np.abs(
            71 / 57600 * w(h_)
            - 71 / 16695 * w2(h_)
            + 71 / 1920 * w3(h_)
            - 17253
            / 339200
            * w4(
                h_,
            )
            + 22 / 525 * w5(h_)
            - 1 / 40 * w6(h_)
        )
        / h_
    )
    while t[-1] < b:
        # update step size until error less than tol
        r = R(h0)
        while r > Tol:
            # the step used to avoid significant step size changing
            sigma = np.clip((Tol / r / 2) ** 0.25, 0.1, 4)
            h0 = min(hmax, sigma * h0)
            # terminate program when step size reach the minimum size
            if h0 < hmin:
                raise Exception("minimum h exceeded !!")
            r = R(h0)
        else:
            yy.append(
                yy[-1]
                + 5179 / 57600 * w(h0)
                + 7571 / 16695 * w2(h0)
                + 393 / 640 * w3(h0)
                - 92097 / 339200 * w4(h0)
                + 187 / 2100 * w5(h0)
                + 1 / 40 * w6(h0)
            )
            # step forward
            t.append(t[-1] + h0)
            h.append(h0)
            # initialize h with new suze
            h0 = b - t[-1]
    return np.asarray(t), np.asarray(yy), np.asarray(h)


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0, 5
    N = 50
    init = 0.5
    t = np.linspace(a, b)

    fig, axes = plt.subplots(
        3, 1, constrained_layout=True, figsize=(10, 10), sharex="col"
    )
    axes[0].plot(t, f_exact(t), "o--", ms=2, label=f"exact value")

    for method in [rk_fehlberg, cash_karp, dormand_prince]:
        t_f, yy_f, h_f = method(f_0, a, b, init, 1e-3, 0.2, Tol=1e-5)
        axes[0].plot(t_f, yy_f, "o", ms=1.5, label=f"{method.__name__} result")
        axes[1].plot(t_f, h_f, "o--", ms=1.5, label=f"{method.__name__} step size")
        err_f = np.abs(yy_f - f_exact(t_f))
        axes[2].plot(t_f, err_f, "o--", ms=1.5, label=f"{method.__name__} error")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    plt.suptitle("$\\frac{\mathrm{d} y}{\mathrm{d} t}=y-t^2+1$")
    plt.show()
