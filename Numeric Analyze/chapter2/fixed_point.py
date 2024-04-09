import matplotlib.pyplot as plt
import numpy as np


def fixedPoint(
    g: callable,
    low,
    high,
    tol,
    maxIter: int = 20,
    return_arr: bool = False,
    verbose: bool = False,
):
    """
    P60
    :param g: function to iterate (after reformat)
    :param low: low bound of the interval
    :param high: upper bound of the interval
    :param tol:  tolerance of the result
    :param maxIter: maximum iteration time
    :param return_arr: if return the points during iteration process
    :param verbose: if print the iteration information
    :return: the final x component of p
    """
    if tol < 0:
        raise Exception("tol should be larger than 0")
    if low > high:
        low, high = high, low
    if g(low) == 0:
        return low
    if g(high) == 0:
        return high
    p0 = low + (high - low) / 2
    arr_p = [p0]
    if g(p0) == 0:
        return np.asarray(arr_p) if return_arr else p0
    for i in range(maxIter):
        p = g(p0)
        arr_p.append(p)
        if verbose:
            print(f"Iteration {i + 1}: {p}")
        if np.allclose(p, p0, atol=tol):
            return np.asarray(arr_p) if return_arr else p
        p0 = p
    else:
        raise Exception("maximum iteration exceed")


def Steffensen(
    g: callable,
    low,
    high,
    tol,
    maxIter: int = 20,
    return_arr: bool = False,
    verbose: bool = False,
):
    """
    P107
    :param g: function to iterate (after reformat)
    :param low: low bound of the interval
    :param high: upper bound of the interval
    :param tol:  tolerance of the result
    :param maxIter: maximum iteration time
    :param return_arr: if return the points during iteration process
    :param verbose: if print the iteration information
    :return: the final x component of p
    """
    if tol < 0:
        raise Exception("tol should be larger than 0")
    if low > high:
        low, high = high, low
    if g(low) == 0:
        return low
    if g(high) == 0:
        return high
    p0 = low + (high - low) / 2
    if g(p0) == 0:
        return np.array([p0]) if return_arr else p0
    arr_p = [p0]
    for i in range(maxIter):
        p1 = g(p0)
        p2 = g(p1)
        if p2 - 2 * p1 + p0 == 0:
            arr_p.append(p2)
            if verbose:
                print(f"Iteration {i + 1}: {p2}")
            return np.asarray(arr_p) if return_arr else p2
        p = p0 - (p1 - p0) ** 2 / (p2 - 2 * p1 + p0)
        arr_p.append(p)
        if verbose:
            print(f"Iteration {i + 1}: {p}")
        if np.allclose(p, p0, atol=tol):
            return np.asarray(arr_p) if return_arr else p
        p0 = p
    else:
        raise Exception("maximum iteration exceed")


def demonstrate(arr_p, f, g, low, high, method="fixed"):
    fix, axes = plt.subplots(
        1, 2, figsize=(10, 6), constrained_layout=True, width_ratios=[6, 4]
    )
    px, py = [], []
    if method[0].lower() == "f":
        for i in range(arr_p.shape[0]):
            px.extend([arr_p[i], g(arr_p[i])])
            py.extend(([g(arr_p[i])] * 2))
        axes[0].set_title("fixed position method")
    elif method[0].lower() == "s":
        for i in range(arr_p.shape[0]):
            p = arr_p[i]
            p1 = g(p)
            p2 = g(p1)
            px.extend([p, p1, p1, p2, p2])
            py.extend([p1, p1, p2, p2, g(p2)])
        axes[0].set_title("Steffensen method")
    else:
        raise Exception("No such method to demonstrate")
    xx = np.linspace(low, high, 100)
    axes[0].axis("equal")
    axes[0].plot(px, py, "ro-", label="iteration points", ms=0.5, lw=0.5)
    axes[0].axline((0, 0), slope=1, label="$y=x$", lw=1, ls="--")
    axes[0].plot(xx, g(xx), "-.", c="orange", label="$g(x)$", lw=1)
    axes[0].plot(xx, f(xx), "g-", label="$f(x)$", lw=1)
    axes[0].plot(arr_p[-1], f(arr_p[-1]), "r*", ms=5, label="root")
    axes[0].axvline(x=arr_p[-1], lw=0.5, ls="-.")
    axes[0].axhline(y=f(arr_p[-1]), lw=0.5, ls="-.")
    axes[0].legend()

    columns = ["$Iteration$", "$p$", "$f(p)$"]
    data = []
    for i in range(arr_p.shape[0]):
        data.append([str(i + 1), str(arr_p[i]), "%.3e" % f(arr_p[i])])
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
    def f(x: np.ndarray):
        return x**3 + 4 * x**2 - 10

    def g(x: np.ndarray):
        # return x - (x ** 3 + 4 * x ** 2 - 10) / (3 * x ** 2 + 8 * x + np.finfo(1e-8).eps)
        return 1 / 2 * np.sqrt(10 - x**3)

    low, high = 1, 2
    arr_p = Steffensen(g, low, high, 1e-5, return_arr=True, verbose=True)
    demonstrate(arr_p, f, g, low, high, "s")


if __name__ == "__main__":
    main()
