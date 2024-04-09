from collections.abc import Sequence
from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from multistep_explicit import (
    adams_bashforth_3,
    adams_bashforth_4,
    adams_bashforth_5,
    milne,
)
from multistep_implicit import (
    adams_moulton_2,
    adams_moulton_3,
    adams_moulton_4,
    simpson_implicit,
)
from runge_kutta import ralston_4


def predictor_corrector(
    f: FunctionType,
    init: Sequence[Real],
    t_pre: Sequence[Real],
    h: int,
    correct: int = 1,
    predictor: FunctionType = milne,
    corrector: FunctionType = simpson_implicit,
):
    """
    :param: correct: the times set to correct the result
    """
    assert len(init) >= 4
    assert h > 0
    # use Adams_Bashforth_4 as predictor
    w = predictor(f, init, t_pre, h)
    # iterate for corrector times
    for _ in range(correct):
        w = corrector(f, init, t_pre, h)(w)
    return w


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1.0
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0.0, 4.0
    N = 50
    h = (b - a) / N
    init = [0.5]
    t = np.linspace(a, b, N + 1)
    for i in range(4):
        init.append(ralston_4(f_0, init[-1], t[i], h))

    fig, axes = plt.subplots(
        2, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    yy_exact = f_exact(t)
    axes[0].plot(t, yy_exact, label="exact value")

    for predic, correct in [
        (adams_bashforth_3, adams_moulton_2),
        (adams_bashforth_4, adams_moulton_3),
        (milne, simpson_implicit),
        (adams_bashforth_5, adams_moulton_4),
    ]:
        yy = init.copy()
        for i in range(5, N + 1):
            yy.append(
                predictor_corrector(
                    f_0, yy, t[:i], h, predictor=predic, corrector=correct
                )
            )
        axes[0].plot(t, yy, "o", ms=2, label=predic.__name__ + "-" + correct.__name__)
        err = np.abs(np.asarray(yy) - yy_exact)
        axes[1].plot(t, err, "o", ms=2, label=predic.__name__ + "-" + correct.__name__)

    axes[0].legend()
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
