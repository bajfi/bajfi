import numpy as np


class TestClass:
    def test_Jacobi(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        Xo, k = iteration.Jacobi(
            a,
            b,
            0,
            maximum_iter=50,
        )
        assert np.allclose(x, Xo, atol=1e-3)

    def test_Gaussian_Seldel(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        Xo, k = iteration.Gaussian_Seidel(
            a,
            b,
            0,
        )
        assert np.allclose(x, Xo, atol=1e-3)

    def test_SOR(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        XO, k = iteration.SOR(
            a,
            b,
            0,
        )
        assert np.allclose(x, XO, atol=1e-3)

    def test_SSOR(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        XO, k = iteration.SSOR(
            a,
            b,
            0,
        )
        assert np.allclose(x, XO, atol=1e-3)

    def test_conjugate_gradient(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        XO, k = conjugate_gradient.conjugate_gradient(
            a,
            b,
            0,
        )
        assert np.allclose(x, XO, atol=1e-3)

    def test_conjugate_gradient_prediction(self):
        a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
        b = np.array([6, 25, -11, 15])
        x = np.array([1, 2, -1, 1])
        prediction = np.diag(np.diag(a) ** -0.5)
        XO, k = conjugate_gradient.conjugate_gradient_precondition(a, b, 0, prediction)
        assert np.allclose(x, XO, atol=1e-3)
