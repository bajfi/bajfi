from numbers import Real
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from multistep_explicit import milne
from multistep_implicit import simpson_implicit
from runge_kutta import ralston_4


def predictor_corrector_adaptive_4(
    f: FunctionType,
    a: Real,
    b: Real,
    init_val: Real,
    hmin: Real,
    hmax: Real,
    RK_method: FunctionType = ralston_4,
    Predictor: FunctionType = milne,
    Corrector: FunctionType = simpson_implicit,
    Sigma: FunctionType = lambda diff, h: 19 * abs(diff) / 270 / h,
    Tol: float = 1e-8,
):
    """
    :param: RK_method is used to generate initial values
    :param: correct: the times set to correct the result
    :param: sigma is the error term between predictor and corrector
    """
    a, b = min(a, b), max(a, b)
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    assert hmax < b - a
    assert hmin > 0
    # initial step size
    h = max((b - a) / 50, hmin)
    # use RK_method to generate next 3 points
    nflag = True  # indicates the value is calculated from RK method
    t = [a]
    # use RK4 to generate the first 3 points
    yy = [init_val]
    for i in range(3):
        yy.append(RK_method(f, yy[-1], t[-1], h))
        t.append(t[-1] + h)
    stepSize = [h] * 4

    while t[-1] < b:
        # use Adams_Bashforth_4 as predictor
        w_p = Predictor(f, yy, t, h)
        # correct result with w_p
        w_c = Corrector(f, yy, t, h)(w_p)
        # adapt step size by the difference between w_c and w_p
        sigma = Sigma(w_c - w_p, h)
        sigma += np.finfo(sigma).eps  # avoid zero divided
        # if result refused, adapt the step size until
        # the error between predictor and corrector
        # is acceptable
        if sigma >= Tol:
            h *= max((Tol / 2 / sigma) ** 0.25, 0.1)
            if h < hmin:
                raise Exception("minimum h exceed !!")
            # if the last 3 points is calculated by RK method
            # update them with new step size
            if nflag:
                for i in range(-3, 0):
                    yy[i] = RK_method(f, yy[i - 1], t[i - 1], h)
                    t[i] = t[i - 1] + h
                    stepSize[i] = h
            # step forward
            else:
                for i in range(3):
                    yy.append(RK_method(f, yy[-1], t[-1], h))
                    t.append(t[-1] + h)
                    stepSize.append(h)
                nflag |= True
        # the branch for accepted
        else:
            yy.append(w_c)
            t.append(t[-1] + h)
            stepSize.append(h)
            nflag &= False
            if t[-1] == b:
                break
            # if sigma is too small or the process is closed to the end
            # we adjust the step size
            if sigma < 0.1 * Tol or t[-1] + h > b:
                h *= min((Tol / 2 / sigma) ** 0.25, 4)
                h = min(hmax, h)
                # the final round we make sure the last point agree with b
                if t[-1] + 4 * h > b:
                    h = abs(b - t[-1]) / 4
                # step forward
                for i in range(3):
                    yy.append(RK_method(f, yy[-1], t[-1], h))
                    t.append(t[-1] + h)
                    stepSize.append(h)
                nflag |= True
    return np.asarray(t), np.asarray(yy), np.asarray(stepSize)


def main():
    f_0 = lambda t_, y_: y_ - t_**2 + 1.0
    f_exact = lambda t_: (t_ + 1) ** 2 - 0.5 * np.exp(t_)
    a, b = 0.0, 4.0
    N = 50
    init = 0.5
    t = np.linspace(a, b, N + 1)

    fig, axes = plt.subplots(
        3, 1, constrained_layout=True, figsize=(8, 8), sharex="col"
    )
    axes[0].plot(t, f_exact(t), "--", label="exact value")

    tt, yy, stepsize = predictor_corrector_adaptive_4(f_0, a, b, init, 1e-5, 0.2)

    axes[0].plot(tt, yy, "o", ms=1.5, label="predictor-corrector-adaptive-4")

    err = np.abs(yy - f_exact(tt))
    axes[1].plot(tt, err, "o--", lw=1, ms=1.5, label="error")

    axes[2].plot(tt, stepsize, "o", ms=1.5, label="step size")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[1].set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
