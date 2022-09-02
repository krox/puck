import numpy as np


def make_wiener(size, T=1.0, rng=None):
    """
    Sample a standard Wiener process on the interval [0,T] on size equidistant points.
        - 'size' can be tuple to create multiple independent paths
        - conventions: result[..., 0] = 0 and variance(result[..., -1]) = T
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(size, int):
        size = (size,)
    assert size[-1] >= 2

    W = rng.standard_normal(size=size)
    W[..., 0] = 0
    W.cumsum(axis=-1, out=W)
    W *= (T/(size[-1]-1))**0.5
    return W


def refine_wiener(coarse, T=1.0, rng=None):
    """
    Refine a Wiener process by subdividing each interval into two. Calling this repeatedly starting from appropriate
    end-points is essentially equivalent to 'make_wiener()'. But sometimes its nice to refine a grid incrementally
    without changing any existing points.
    """
    if rng is None:
        rng = np.random.default_rng()
    coarse = np.array(coarse)
    assert coarse.ndim >= 1 and coarse.shape[-1] >= 2

    eta = rng.standard_normal(coarse.shape[:-1] + (coarse.shape[-1]-1,))
    eta *= (T/(coarse.shape[-1]-1))**0.5

    fine = np.zeros(coarse.shape[:-1]+(coarse.shape[-1]*2-1,))
    fine[..., ::2] = coarse
    fine[..., 1::2] = 0.5*(coarse[..., 1:] + coarse[..., :-1] + eta)
    return fine

