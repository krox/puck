""" simulate autoregressive models """

import numpy as np


def run_autoregressive(phi, steps, runs=None, mu=0.0, sigma=1.0, seed=None):
    """
    run a stochastic process based on the AR(p) model
    x[t] = phi[0]*x[t-1] + ... + phi[p-1]*x[t-p] + eta[t]
    with eta[t] gaussian with parameters mu,sigma and x[t] = 0 for t<0
     - for best performance, run multiple runs in parallel
     - returns array of shape (runs, steps), or (steps,) if runs==None
    """
    phi = np.array(phi, ndmin=1)
    assert phi.ndim == 1
    rng = np.random.RandomState(seed=seed)
    x = np.zeros((1 if runs is None else runs, steps))
    for t in range(steps):
        q = np.min([t, phi.shape[0]])
        x[:, t] = x[:, t - q:t].dot(phi[:q][::-1]) + \
            rng.randn(1 if runs is None else runs) * sigma + mu

    if runs is None:
        return x[0, :]
    else:
        return x
