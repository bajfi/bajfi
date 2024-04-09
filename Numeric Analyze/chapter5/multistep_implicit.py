import os
from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

os.path.join("../chapter2/")
from chapter2.newton import secant
from types import LambdaType

"""
f: dy/dt = f(t,y)
h: time step
init: previous values
t: previous time
cur_t: current time
"""


def adams_moulton_2(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
) -> LambdaType:
    assert h > 0
    assert len(init) >= 2
    assert len(t_pre) >= 2
    return lambda y: init[-1] + h / 12 * (
        5 * f(t_pre[-1] + h, y) + 8 * f(t_pre[-1], init[-1]) - f(t_pre[-2], init[-2])
    )


def adams_moulton_3(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
) -> LambdaType:
    assert h > 0
    assert len(init) >= 3
    assert len(t_pre) >= 3
    return lambda y: init[-1] + h / 24 * (
        9 * f(t_pre[-1] + h, y)
        + 19 * f(t_pre[-1], init[-1])
        - 5 * f(t_pre[-2], init[-2])
        + 1 * f(t_pre[-3], init[-3])
    )


def adams_moulton_4(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
) -> LambdaType:
    assert h > 0
    assert len(init) >= 4
    assert len(t_pre) >= 4
    return lambda y: init[-1] + h / 720 * (
        251 * f(t_pre[-1] + h, y)
        + 646 * f(t_pre[-1], init[-1])
        - 264 * f(t_pre[-2], init[-2])
        + 106 * f(t_pre[-3], init[-3])
        - 19 * f(t_pre[-4], init[-4])
    )


def simpson_implicit(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
) -> LambdaType:
    assert h > 0
    assert len(init) >= 2
    assert len(t_pre) >= 2
    return lambda y: init[-2] + h / 3 * (
        f(t_pre[-1] + h, y) + 4 * f(t_pre[-1], init[-1]) + f(t_pre[-2], init[-2])
    )


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1.0
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0, 4
    N = 50
    h = (b - a) / N
    init = 0.5
    t = np.linspace(a, b, N + 1)
    yy = f_exact(t)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    axes[0].plot(t, f_exact(t), label="exact value")

    for i in range(2, 5):
        f = eval("adams_moulton_" + f"{i}")
        init_val = f_exact(t[:i]).tolist()
        # we use Secant method to get the root
        for j in range(i, t.shape[0]):
            # the function to find root
            f_eval = lambda y: f(f_0, init_val, t[:j], h)(y) - y
            init_val.append(secant(f_eval, 0, 6, tol=1e-8, maxIter=50))
        axes[0].plot(t, init_val, "o", ms=2, label="adams_moulton_" + f"{i}")
        err = np.abs(np.asarray(init_val) - yy)
        axes[1].plot(t, err, "o--", ms=2, label="adams_moulton_" + f"{i}")

    for _ in range(1):
        init_val = f_exact(t[:2]).tolist()
        # we use Secant method to get the root
        for j in range(2, t.shape[0]):
            # the function to find root
            f_eval = lambda y: simpson_implicit(f_0, init_val, t[:j], h)(y) - y
            init_val.append(secant(f_eval, 0, 6, tol=1e-8, maxIter=50))
        axes[0].plot(t, init_val, "o", ms=2, label="simpson implicit")
        err = np.abs(np.asarray(init_val) - yy)
        axes[1].plot(t, err, "o--", ms=2, label="simpson implicit")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
