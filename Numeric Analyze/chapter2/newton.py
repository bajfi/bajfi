from numbers import Real
from types import FunctionType
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def newton(
    f: FunctionType,
    f1: FunctionType,
    x0: Real,
    *,
    f2: FunctionType = None,
    tol: Real = 1e-5,
    maxIter: int = 20,
    return_arr: bool = False,
    verbose: bool = False,
):
    """
    P68
    :param f: original function to solve
    :param x0: initial guess
    :param f1:first order directive of f
    :param f2:second order directive of f
    :param tol:  tolerance of the result
    :param maxIter: maximum iteration time
    :param return_arr: if return the points during iteration process
    :param verbose: if print the iteration information
    :return: the final x component of p
    """
    assert tol > 0
    if f1 is None:
        raise Exception("f1 function should be input")
    arr_x: List[Real] = [x0]
    if f(x0) == 0:
        return np.asarray(arr_x) if return_arr else x0
    if f1(x0) == 0:
        raise ValueError("f'(p0) == 0, can't continue,try change root interval")
    for i in range(maxIter):
        x = x0 - (
            f(x0) / f1(x0)
            if f2 is None
            else f(x0) * f1(x0) / (f1(x0) ** 2 - f(x0) * f2(x0))
        )
        arr_x.append(x)
        if verbose:
            print(f"Iteration {i + 1}: {x}")
        if np.allclose(x, x0, atol=tol):
            return np.asarray(arr_x) if return_arr else x
        x0 = x
    else:
        raise Exception("maximum iteration exceed")


def secant(
    f: FunctionType,
    x0: Real,
    x1: Real,
    tol: float = 1e-5,
    maxIter: int = 30,
    return_arr: bool = False,
    verbose: bool = False,
):
    """
    P72
    :param f: original function to solve
    :param x0: first guess
    :param x1: second guess
    :param tol:  tolerance of the result
    :param maxIter: maximum iteration time
    :param return_arr: if return the points during iteration process
    :param verbose: if print the iteration information
    :return: the final x component of p
    """
    if tol < 0:
        raise Exception("tol should be larger than 0")
    if f(x0) == 0:
        return np.array([x0]) if return_arr else x0
    if f(x1) == 0:
        return np.array([x0, x1]) if return_arr else x1
    arr_x: List[Real] = [x0, x1]
    for i in range(maxIter):
        x = (
            x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            if (f(x1) - f(x0)) != 0
            else (x0 + (x1 - x0) / 2)
        )
        arr_x.append(x)
        if verbose:
            print(f"Iteration {i + 2}: {x}")
        if abs(x - x1) < tol:
            return np.asarray(arr_x) if return_arr else x
        x0, x1 = x1, x
    else:
        raise Exception("maximum iteration exceed")


def false_position(
    f: FunctionType,
    x0: Real,
    x1: Real,
    tol: Real = 1e-5,
    maxIter: int = 30,
    return_arr: bool = False,
    verbose: bool = False,
):
    """
    P74
    :param f: original function to solve
    :param x0: first initial guess
    :param x1: second initial guess
    :param tol:  tolerance of the result
    :param maxIter: maximum iteration time
    :param return_arr: if return the points during iteration process
    :param verbose: if print the iteration information
    :return: the final x component of p
    """
    if tol < 0:
        raise Exception("tol should be larger than 0")
    if np.sign(f(x0)) == np.sign(f(x1)):
        raise Exception("low and high bound should be sign different")
    if f(x0) == 0:
        return np.array([x0]) if return_arr else x0
    if f(x1) == 0:
        return np.array([x0, x1]) if return_arr else x1
    arr_x: List[Real] = [x0, x1]
    for i in range(maxIter):
        x = (
            x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
            if (f(x1) - f(x0)) != 0
            else (x0 + (x1 - x0) / 2)
        )
        arr_x.append(x)
        if verbose:
            print(f"Iteration {i + 2}: {x}")
        if np.allclose(x, x1, atol=tol):
            return np.asarray(arr_x) if return_arr else x
        x0, x1 = (x, x1) if np.sign(f(x)) != np.sign(f(x1)) else (x0, x)
    else:
        raise Exception("maximum iteration exceed")


def demonstrate(arr_x, f, method="newton"):
    farr_p = f(arr_x)
    px, py = [], []
    fix, axes = plt.subplots(
        1, 2, figsize=(10, 6), constrained_layout=True, width_ratios=[6, 4]
    )
    if method[0].lower() == "n":
        px.append(arr_x[0])
        py.append(farr_p[0])
        for i in range(1, arr_x.shape[0]):
            px.extend([arr_x[i]] * 2)
            py.extend([0, f(arr_x[i])])
        axes[0].set_title("Newton Iteration")
    elif method[0].lower() == "s":
        for i in range(arr_x.shape[0] - 2):
            px.extend([arr_x[i], arr_x[i + 1], arr_x[i + 2], arr_x[i + 2], np.nan])
            py.extend([farr_p[i], farr_p[i + 1], 0, farr_p[i + 2], np.nan])
        axes[0].set_title("Secant Iteration")
    elif method[0].lower() == "f":
        px.extend([arr_x[0], arr_x[1], np.nan])
        py.extend([farr_p[0], farr_p[1], np.nan])
        for i in range(arr_x.shape[0] - 2):
            px.extend(
                [px[-3], arr_x[i + 2], np.nan]
                if np.sign(py[-3]) != np.sign(farr_p[i + 2])
                else [px[-2], arr_x[i + 2], np.nan]
            )
            py.extend(
                [py[-3], farr_p[i + 2], np.nan]
                if np.sign(py[-3]) != np.sign(farr_p[i + 2])
                else [py[-2], farr_p[i + 2], np.nan]
            )
        for i in range(arr_x.shape[0] - 2):
            px.extend([arr_x[i + 2], arr_x[i + 2], np.nan])
            py.extend([farr_p[i + 2], 0, np.nan])
        axes[0].set_title("False Position Iteration")
    else:
        raise Exception("No such method to demonstrate")
    xx = np.linspace(min(arr_x.min(), min(px)), max(arr_x.max(), max(px)), 100)
    axes[0].axis("equal")
    axes[0].plot(xx, f(xx), "g-", label="$f(x)$", lw=1)
    axes[0].axhline(0, label="$y=0$", lw=1, ls="--")
    axes[0].plot(px, py, "ro-", label="iteration points", ms=0.8, lw=0.5)
    axes[0].legend()

    columns = ["$Iteration$", "$p$", "$f(p)$"]
    data = []
    for i in range(arr_x.shape[0]):
        data.append([str(i + 1), str(arr_x[i]), "%.3e" % f(arr_x[i])])
    axes[1].table(
        data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.5, 0.3],
    )
    axes[1].axis("off")
    plt.show()


def main():
    def f(x):
        return np.exp(x) - x - 1
        # return np.log(x) + np.cos(x / 2) - np.sqrt(x) + .2

    def f1(x):
        return np.exp(x) - 1
        # return 1 / x - .5 * np.sin(x / 2) - .5 / np.sqrt(x)

    def f2(x):
        return np.exp(x)

    # arr_x = newton(f, f1, f2=f2, x0=-1, tol=1e-5, maxIter=20, return_arr=True,
    #                verbose=True)
    arr_x = secant(
        f, x0=-1, x1=0.5, tol=1e-5, maxIter=40, return_arr=True, verbose=True
    )
    demonstrate(arr_x, f, method="scant")


if __name__ == "__main__":
    main()
