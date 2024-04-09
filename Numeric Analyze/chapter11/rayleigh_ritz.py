from numbers import Real
from types import FunctionType


def peicewise_linear_rayleigh_ritz(
    px: FunctionType, qx: FunctionType, fx: FunctionType, a: Real, b: Real, N, int
):
    assert N > 1
    a, b = min(a, b), max(a, b)
    h: Real = (b - a) / N
    return
