""" binning/bootstrap/reweighting """

import numpy as np


def binned(data, rwt=None, binsize=None, nbins=None):
    """bin data along axis=0"""
    assert (binsize is None) or (nbins is None)

    if binsize is not None:
        nbins = data.shape[0] // binsize
    if nbins is not None:
        binsize = data.shape[0] // nbins
    else:
        nbins = np.min([100, data.shape[0]])
        binsize = data.shape[0] // nbins

    data = data[:nbins * binsize].reshape((nbins, binsize, *data.shape[1:]))

    if rwt is None:
        return data.mean(axis=1)
    else:
        rwt = rwt[:nbins * binsize].reshape((nbins, binsize))
        data = (rwt.reshape((nbins, binsize, *((1,) * (len(data.shape) - 2)))) * data[:]).mean(axis=1)
        rwt = rwt.mean(axis=1)
        return data, rwt


def bootstrap_mean(data, rwt=None, samples=100, seed=0):
    """create bootstrap samples along axis=0 and compute sample mean"""
    rng = np.random.RandomState(seed=seed)
    n = data.shape[0]
    ws = 1.0 * rng.multinomial(n, [1.0 / n] * n, size=samples)

    # NOTE: Maybe putting the reweighting into the probabilities of the multinomial would be clever...
    if rwt is not None:
        assert rwt.shape[0] == ws.shape[1]
        ws = ws * rwt
    ws *= 1.0 / ws.sum(axis=1, keepdims=True)
    return np.tensordot(ws, data, axes=(1, 0))


def bootstrap_var(data, samples=100, seed=0):
    """create bootstrap samples along axis=0 and compute sample .var"""
    bs = np.zeros((samples, *data.shape[1:]), dtype=data.dtype)
    rng = np.random.RandomState(seed=seed)
    for k in range(samples):
        c = rng.choice(data.shape[0], data.shape[0])
        bs[k] = data[c].var(axis=0)
    return bs


def bootstrap_std(data, samples=100, seed=0):
    """create bootstrap samples along axis=0 and compute sample .std"""
    bs = np.zeros((samples, *data.shape[1:]), dtype=data.dtype)
    rng = np.random.RandomState(seed=seed)
    for k in range(samples):
        c = rng.choice(data.shape[0], data.shape[0])
        bs[k] = data[c].std(axis=0)
    return bs


def correlation(a, b):
    """(normalized) correlation coefficient between two datasets of same size"""

    assert a.shape == b.shape

    a = np.copy(a)
    b = np.copy(b)
    a -= a.mean()
    b -= b.mean()
    a /= a.std()
    b /= b.std()
    return (a * b).mean()


def check_distribution(data, dist, bins=50):
    """ plot a histogram and draw a reference distribution (automatic normalization) """

    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(data, bins=bins, density=True)
    xs = np.linspace(np.min(data), np.max(data), 500)
    Z = quad(dist, xs[0], xs[-1])[0]
    plt.plot(xs, dist(xs) / Z)
