import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

from iteration import gaussian_seidel, jacobi, sor, ssor


def main():
    R = 1
    Center = np.array([1.0, 1.0])
    theta = np.linspace(0, 2 * np.pi, 100)
    transformer = np.array([[1.0, 0.5], [0.3, 1.0]]).T

    fig, axes = plt.subplots(
        1, 2, constrained_layout=True, figsize=(12, 6), sharex="col"
    )

    xx_ori = R * np.cos(theta) + Center[0]
    yy_ori = R * np.sin(theta) + Center[1]
    points = np.asarray([xx_ori, yy_ori])

    axes[0].plot(xx_ori, yy_ori, "o--", ms=2, label="origin shape")
    axes[0].plot(Center[0], Center[1], "o", ms=5, label="origin center")

    points_trans = transformer @ points
    axes[0].plot(points_trans[0], points_trans[1], "o--", ms=2, label="transform shape")
    Center_trans = transformer @ Center
    axes[0].plot(Center_trans[0], Center_trans[1], "o", ms=5, label="transform center")

    for i in range(points.shape[1]):
        axes[0].plot(
            [points[0][i], points_trans[0][i]],
            [points[1][i], points_trans[1][i]],
            "r--",
        )

    # =========================================================================

    inv_transform = linalg.inv(transformer)
    Center_trans_inv = inv_transform @ Center
    axes[1].plot(
        Center_trans_inv[0], Center_trans_inv[1], "ro", ms=5, label="target point"
    )

    Tol = R / 10
    xx_tol = Tol * np.cos(theta) + Center_trans_inv[0]
    yy_tol = Tol * np.sin(theta) + Center_trans_inv[1]
    points_tol = np.asarray([xx_tol, yy_tol])
    axes[1].plot(xx_tol, yy_tol, "o--", ms=1, label="tolerance region")

    points_tol_trans = transformer @ points_tol
    axes[1].plot(
        points_tol_trans[0],
        points_tol_trans[1],
        "o--",
        ms=2,
        label="tolerance region transform",
    )

    for method in [jacobi, gaussian_seidel, sor, ssor]:
        move_points = method(transformer, Center, 0, return_points=True)
        move_points = np.asarray(move_points).T
        axes[1].plot(move_points[0], move_points[1], "o--", ms=2, label=method.__name__)

    axes[0].legend()
    axes[1].legend()
    axes[1].axis("equal")
    plt.suptitle(f"condition number is {linalg.cond(transformer).__round__(2)}")
    plt.show()


if __name__ == "__main__":
    main()
