from collections.abc import Sequence
from numbers import Real

import numpy as np
from numpy import ndarray

from chapter6.gaussian_elimination import guass_eliminate


def gram_schmidt(vectors: Sequence[Sequence[Real]], normalize: bool = False):
    """
    take it as row vector
    """
    vectors: ndarray = np.asarray(
        vectors,
        dtype=np.float_,
    )
    assert vectors.ndim == 2
    # use Gaussian elimination to check if it's linear independent
    tmp: ndarray = np.array(vectors)
    try:
        guass_eliminate(tmp)
    except Exception:
        raise ValueError("vectors should be linear independent")
    # Gram Schmidt process
    mag: ndarray = np.empty((vectors.shape[0], 1))
    mag[0][0] = np.dot(vectors[0], vectors[0])
    for i in range(1, vectors.shape[0]):
        vectors[i] -= sum(
            np.dot(vectors[j], vectors[i]) / mag[j][0] * vectors[j] for j in range(i)
        )
        mag[i][0] = np.dot(vectors[i], vectors[i])
    return vectors / mag**0.5 if normalize else vectors


if __name__ == "__main__":
    vectors = [[1, 0, 2], [1, 3, 1], [2, 1, 1]]
    vector_g = gram_schmidt(vectors, normalize=True)
    vector_g_inv = np.linalg.inv(vector_g)
    print(vector_g)
    print(vector_g_inv)
