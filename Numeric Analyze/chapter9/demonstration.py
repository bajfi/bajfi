import matplotlib.pyplot as plt
import numpy as np

"""
demonstration for Power method
"""


def main():
    A = np.array([[-2.0, -3], [6, 7]])
    x = np.array([1.0, 1])
    # eigenvalue of A is 4 and 1
    # eigen vectors are [1,-2],[1,-1]
    eig, eig_vec = np.linalg.eig(A)
    print(eig)
    print(eig_vec)

    for i in range(10):
        xi = np.linalg.matrix_power(A, i) @ x
        print(xi)
        xi /= np.abs(xi).max()
        x = xi
        plt.plot([0, xi[0]], [0, xi[1]], label=f"transform {i}")

    plt.plot([0, 1], [0, -2], label="eigen vector")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
