import matplotlib.pyplot as plt
import numpy as np


def parameter_curve(x0, y0, a0, b0, x1, y1, a1, b1):
    """
    left guide point : (x+a0,y+b0)
    right guide point : (x-a1,y-b1)
    """
    t = np.linspace(0, 1, 50)
    xt = (
        (2 * (x0 - x1) + (a0 + a1)) * t**3
        + (3 * (x1 - x0) - (a1 + 2 * a0)) * t**2
        + a0 * t
        + x0
    )
    yt = (
        (2 * (y0 - y1) + (b0 + b1)) * t**3
        + (3 * (y1 - y0) - (b1 + 2 * b0)) * t**2
        + b0 * t
        + y0
    )
    return xt, yt


def main():
    x0, y0 = 0, 0
    x1, y1 = 1, 0
    a0, b0 = 1, 1
    a1, b1 = 1, -1
    xt, yt = parameter_curve(x0, y0, a0, b0, x1, y1, a1, b1)
    plt.plot(xt, yt, label="Bezier curve")
    plt.plot([x0 + a0, x0], [y0 + b0, y0], [x1 - a1, x1], [y1 - b1, y1], "r-")
    plt.plot([x0 + a0, x1 - a1], [y0 + b0, y1 - b1], "ro", ms=5, label="guide points")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
