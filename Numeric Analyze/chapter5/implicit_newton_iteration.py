from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from runge_kutta import ralston_4, rk4


def trapezoidal_newton(
    f: FunctionType,
    f_y: FunctionType,
    init: Real,
    t_pre: Real,
    h: Real,
    Tol: Real = 1e-8,
    maxIter: int = 50,
) -> np.ndarray:
    assert h > 0
    assert Tol > 0
    # get the function used for iteration
    F = lambda y: y - init - 0.5 * h * (f(t_pre + h, y) + f(t_pre, init))
    # set up a initial point and start iterating
    w0 = init + 0.5 * h * f(t_pre, init)
    for i in range(maxIter):
        w1 = w0 - F(w0) / (1 - 0.5 * h * f_y(t_pre + h, w0))
        if abs(w0 - w1 < Tol):
            return w1
        w0 = w1
    else:
        raise Exception("maximum iteration exceeded !!")


def main():
    # use a stiff function to compare trapezoidal_Newton and RK method
    f_0 = lambda t_, y_: 5 * np.exp(5 * t_) * (y_ - t_) ** 2 + 1
    f_y = lambda t_, y_: 10 * np.exp(5 * t_) * (y_ - t_)
    f_exact = lambda t_: t_ - np.exp(-5 * t_)
    a, b = 0.0, 4.0
    N = 20
    h = (b - a) / N
    init = -1.0
    t = np.linspace(a, b, N + 1)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    yy_exact = f_exact(t)
    axes[0].plot(t, yy_exact, "--", label="exact value")

    for method in [rk4, ralston_4]:
        yy = [init]
        for i in range(N):
            yy.append(method(f_0, yy[-1], t[i], h))
        axes[0].plot(t, yy, "o", ms=2, label=method.__name__)
        err = np.abs(np.asarray(yy) - yy_exact)
        axes[1].plot(t, err, "o", ms=2, label=method.__name__)

    for method in [trapezoidal_newton]:
        yy = [init]
        for i in range(N):
            yy.append(method(f_0, f_y, yy[-1], t[i], h))
        axes[0].plot(t, yy, "o", ms=2, label=method.__name__)
        err = np.abs(np.asarray(yy) - yy_exact)
        axes[1].plot(t, err, "o", ms=2, label=method.__name__)

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
