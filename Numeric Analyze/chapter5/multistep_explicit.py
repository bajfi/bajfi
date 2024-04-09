from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

"""
f: dy/dt = f(t,y)
h: time step
t:current time
"""


def adams_bashforth_2(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
):
    assert h > 0
    assert len(init) >= 2
    assert len(t_pre) >= 2
    return init[-1] + h / 2 * (3 * f(t_pre[-1], init[-1]) - f(t_pre[-2], init[-2]))


def adams_bashforth_3(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
):
    assert h > 0
    assert len(init) >= 3
    assert len(t_pre) >= 3
    return init[-1] + h / 12 * (
        23 * f(t_pre[-1], init[-1])
        - 16 * f(t_pre[-2], init[-2])
        + 5 * f(t_pre[-3], init[-3])
    )


def adams_bashforth_4(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
):
    assert h > 0
    assert len(init) >= 4
    assert len(t_pre) >= 4
    return init[-1] + h / 24 * (
        55 * f(t_pre[-1], init[-1])
        - 59 * f(t_pre[-2], init[-2])
        + 37 * f(t_pre[-3], init[-3])
        - 9 * f(t_pre[-4], init[-4])
    )


def adams_bashforth_5(
    f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real
):
    assert h > 0
    assert len(init) >= 5
    assert len(t_pre) >= 5
    return init[-1] + h / 720 * (
        1901 * f(t_pre[-1], init[-1])
        - 2774 * f(t_pre[-2], init[-2])
        + 2616 * f(t_pre[-3], init[-3])
        - 1274 * f(t_pre[-4], init[-4])
        + 251 * f(t_pre[-5], init[-5])
    )


def milne(f: FunctionType, init: Sequence[Real], t_pre: Sequence[Real], h: Real):
    """
    occasionally used as a predictor for simpson implicit method
    """
    assert h > 0
    assert len(init) >= 4
    assert len(t_pre) >= 4
    return init[-4] + 4 * h / 3 * (
        2 * f(t_pre[-1], init[-1]) - f(t_pre[-2], init[-2]) + 2 * f(t_pre[-3], init[-3])
    )


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0, 2
    N = 50
    h = (b - a) / N
    init = 0.5
    t = np.linspace(a, b, N + 1)
    yy = f_exact(t)

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    axes[0].plot(t, f_exact(t), label="exact value")

    for i in range(2, 6):
        f = eval("adams_bashforth_" + f"{i}")
        init_val = f_exact(t[:i]).tolist()
        for j in range(i, t.shape[0]):
            init_val.append(f(f_0, init_val, t[:j], h))
        axes[0].plot(t, init_val, "o", ms=2, label="adams_bashforth_" + f"{i}")
        err = np.abs(np.asarray(init_val) - yy)
        axes[1].plot(t, err, "o--", ms=2, label="adams_bashforth_" + f"{i}")

    for _ in range(1):
        f = eval("milne")
        init_val = f_exact(t[:4]).tolist()
        for j in range(4, t.shape[0]):
            init_val.append(f(f_0, init_val, t[:j], h))
        axes[0].plot(t, init_val, "o", ms=2, label="milne")
        err = np.abs(np.asarray(init_val) - yy)
        axes[1].plot(t, err, "o--", ms=2, label="milne")

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
