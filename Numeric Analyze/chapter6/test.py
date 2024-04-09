import numpy as np
from scipy import sparse

from chapter6 import factorization, substitution


class TestClass:
    def test_forwardSubstitution(self):
        trilMatrix = np.array(
            [[1, 0, 0, 0, 8], [2, 1, 0, 0, 7], [3, 4, 1, 0, 14], [-1, -3, 0, 1, -7]]
        )
        ans = np.array([8, -9, 26, -26])
        assert np.allclose(substitution.forward_substitution(trilMatrix), ans)

    def test_backSubstitution(self):
        triuMatrix = np.array(
            [
                [1, 1, 0, 3, 8],
                [0, -1, -1, -5, -9],
                [0, 0, 3, 13, 26],
                [0, 0, 0, -13, -26],
            ]
        )
        ans = np.array([3, -1, 0, 2])
        assert np.allclose(substitution.backward_substitution(triuMatrix), ans)

    def test_LU(self):
        a = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
        L, U = factorization.lu(a)
        assert np.allclose(L @ U, a)

    def test_LDLdecomposition(self):
        a = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
        L, D = factorization.ldl(a)
        assert np.allclose(L @ sparse.diags(D) @ L.T, a)

    def test_Cholesky(self):
        a = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
        L = factorization.cholesky(a)
        assert np.allclose(L @ L.T, a)

    def test_Crout(self):
        a = np.array(
            [[2, -1, 0, 0, 1], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, 1]]
        )
        arr = np.hsplit(
            a,
            [
                a.shape[0],
            ],
        )
        x = factorization.crout(a)
        b = arr[1].squeeze()
        assert np.allclose(arr[0] @ x, b)
