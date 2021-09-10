""" analyze (and plot) autocorrelation of time-series """

import numpy as np
from puck.linear_fitting import fit_linear_basic


def pad_axis(a, axis):
    """ helper to pad an array in a specific axis """
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (0, a.shape[axis])
    return np.pad(a, pad_width=pad_width, mode="constant", constant_values=0)


def acf(a, common_mean=False, naive=False):
    """
    (normalized) autocorrelation function and errors using Bartlett's formula.
    For multidimensional arrays, compute autocorrelation along last axis.
    Separate compuations along orthogonal axes.

      * common_mean=True assumes the same mean for all parallel series
        (which might improve accuracy)
    """

    # normalize (also does a copy)
    if common_mean:
        a = a - a.mean(keepdims=True)
    else:
        a = a - a.mean(axis=-1, keepdims=True)

    # correlation coefficients
    a = pad_axis(a, axis=-1)
    a = np.fft.fft(a, axis=-1)
    a = a * a.conj()
    a = np.fft.ifft(a, axis=-1)
    a = a[..., :a.shape[-1] // 2]
    for i in range(a.shape[-1]):
        a[..., i] /= a.shape[-1] - i
    a = a.real / a.real[..., np.newaxis, 0]

    # error bars
    e = np.zeros(a.shape)
    e[..., 0] = 0.0
    e[..., 1] = 1.0 / a.shape[-1]
    e[..., 2:] = (1.0 + 2 * np.cumsum(a[..., 1:-1]**2, axis=-1)) / a.shape[-1]

    return (a, np.sqrt(e))


def autocorrelation(a, scale=1.0, C=5.0, common_mean=False, plot=False, log_plot=False, naive=False):
    """
    integrated autocorrelation time using standard formula with cutoff.
    For multidimensional arrays, compute autocorrelation along last axis.
    Separate calculations along orthogonal axes.

      * common_mean=True assumes the same mean for all parallel series
        (which might improve accuracy)
      * naive=True uses the basic formula for correlations (instead of the FFT
        based one). For the general case this is slower, but for very long
        series, where only a few correlation coefficients are needed, this might
        be faster (and uses less memory)
    """

    assert a.ndim >= 1

    if naive:
        if common_mean:
            tmp = a - a.mean(keepdims=True)
        else:
            tmp = a - a.mean(axis=-1, keepdims=True)
        tmp = tmp.reshape((-1, tmp.shape[-1]))
        tau = np.zeros(tmp.shape[0])
        for k in range(len(tau)):
            tau[k] = 0.5
            c0 = tmp[k].dot(tmp[k]) / tmp.shape[-1]
            lag = 1
            while lag <= C * tau[k] and lag < tmp.shape[-1]:
                tau[k] += tmp[k, lag:].dot(tmp[k, :-lag]) / ((tmp.shape[-1]-lag) * c0)
                lag += 1
        tau = tau.reshape(a.shape[:-1])

    else:
        # correlation coefficients and errors
        c, e = acf(a, common_mean=common_mean)
        assert c.shape == a.shape

        # tau = 0.5 + sum C_t
        c = c.reshape((-1, a.shape[-1]))
        tau = np.zeros(c.shape[0])
        for k in range(len(tau)):
            tau[k] = 0.5
            i = 1
            while i <= C * tau[k] and i < c.shape[-1]:
                tau[k] += c[k, i]
                i += 1
        c = c.reshape(a.shape)
        tau = tau.reshape(a.shape[:-1])

    if tau.ndim == 0:
        tau = tau[()]

    if plot:
        assert not naive  # the naive algo does not produce error bars and such (could be implemented of course)

        if plot is True:
            plot = plt
            plt.figure(figsize=(12, 4))
        assert a.ndim == 1
        import matplotlib.pyplot as plt

        if log_plot:
            plot.semilogy()
        w = int(C * tau) + 1
        plot.errorbar(np.arange(w) * scale, c[:w], yerr=e[:w], fmt="x")
        xs = np.linspace(0, C * tau)
        plot.errorbar(xs * scale, np.exp(-xs / tau), fmt="-", label=r"$\tau_{int} = "+f"{tau*scale:.3f}$")
        plot.grid()
        plot.legend()

    return tau * scale


def autocorrelation_ext(a, plot=False, scale=1.0, C=5.0, log_plot=False):
    """autocorrelation using a 2-exponential fit"""
    assert len(a.shape) == 1

    # correlation coefficients and errors
    c, e = acf(a)

    tau = 0.5
    i = 1
    while i <= C * tau and i < a.shape[0]:
        tau += c[i]
        i += 1

    w = int(C * tau) + 1
    xs = np.arange(1, w)

    def f(tau1, tau2):
        # a1,a2 = linear_fit_basic(c[1:w], e[1:w], [np.exp(-xs/tau1), np.exp(-xs/tau2)])
        # two-exponent fit stabilized using Thomas' trick
        a1, a2 = linear_fit_basic(c[1:w], e[1:w], [
                                    np.exp(-xs / tau1), np.heaviside(w // 2 - xs, 0.0) * np.exp(-xs / tau2)])
        chi2 = np.sum(
            ((a1 * np.exp(-xs / tau1) + a2 * np.exp(-xs / tau2) - c[1:w]) / e[1:w])**2)
        return chi2

    from iminuit import Minuit
    m = Minuit(f, tau1=0.8 * tau, tau2=1.2 * tau)
    m.errordef = 1
    m.errors["tau1"] = np.abs(tau)*0.1
    m.errors["tau2"] = np.abs(tau)*0.1
    m.migrad()
    tau1 = m.values["tau1"]
    tau2 = m.values["tau2"]
    a1, a2 = linear_fit_basic(
        c[1:w], e[1:w], [np.exp(-xs / tau1), np.exp(-xs / tau2)])
    if tau1 > tau2:
        tau1, tau2 = tau2, tau1
        a1, a2 = a2, a1

    tau = 0.5 + a1 * (1 / (1 - np.exp(-1 / tau1)) - 1) + \
        a2 * (1 / (1 - np.exp(-1 / tau2)) - 1)

    if plot:
        import matplotlib.pyplot as plt

        if plot is True:
            plot = plt
            plt.figure(figsize=(12, 4))

        w = int(C * tau) + 1
        plot.errorbar(np.arange(w) * scale, c[:w], e[:w], fmt="x")
        xs = np.linspace(0, C * tau)

        plot.errorbar(xs * scale, a1 * np.exp(-xs / tau1) + a2 * np.exp(-xs / tau2), fmt="-", label=f"tau = {tau*scale:.3f}, tau1 = {tau1*scale:.3f}, tau2 = {tau2*scale:.3f}")

        if log_plot:
            plot.semilogy()
        plot.grid()
        plot.legend()

    # return tau1*scale, tau2*scale, a1, a2
    return tau


def autocorrelation_ar1(a, plot=False, scale=1.0, C=5.0, log_plot=False):
    """integrated autocorrelation time using a AR(1) model"""
    assert a.ndim == 1

    a = a - a.mean()
    phi = a[:-1].dot(a[1:]) / a[:-1].dot(a[:-1])
    tau = 1. / (1. - phi) - 0.5

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        c, e = acf(a)
        w = int(tau * C) + 1
        plt.errorbar(np.arange(w) * scale, c[:w], e[:w], fmt="x")
        xs = np.linspace(0, w)
        plt.errorbar(xs * scale, np.power(phi, xs), fmt="-", label=f"tau = {tau*scale:.3f}")

        if log_plot:
            plt.semilogy()
        plt.grid()
        plt.legend()

    return tau * scale


def autocorrelation_ar2(a, plot=False, scale=1.0, C=5.0, log_plot=False):
    """integrated autocorrelation time using a AR(2) model"""
    assert a.ndim == 1

    a = a - a.mean()

    M = np.zeros((2, 2))
    r = np.zeros((2,))
    M[0, 0] = a[1:-1].dot(a[1:-1])
    M[0, 1] = a[0:-2].dot(a[1:-1])
    M[1, 0] = a[1:-1].dot(a[0:-2])
    M[1, 1] = a[0:-2].dot(a[0:-2])
    r[0] = a[2:].dot(a[1:-1])
    r[1] = a[2:].dot(a[0:-2])
    phi1, phi2 = np.linalg.solve(M, r)
    tau = 0.5 * (phi1 * phi2 - phi2**2 + phi1 + 1) / \
        (phi1 * phi2 + phi2**2 - phi1 - 2 * phi2 + 1)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        c, e = acf(a)
        w = int(tau * C) + 1
        plt.errorbar(np.arange(w) * scale, c[:w], e[:w], fmt="x")

        # lamb2 is often negative, which is problematic for plotting
        # xs=np.linspace(0.1, w)
        xs = np.arange(w + 1)

        d = np.sqrt(phi1**2 + 4 * phi2)
        lamb1 = 0.5 * (phi1 + d)
        lamb2 = 0.5 * (phi1 - d)
        print(f"d={d}")
        print(f"lamb1={lamb1}")
        print(f"lamb2={lamb2}")
        print(f"phi1 = {phi1}")
        print(f"phi2 = {phi2}")

        c0 = 1.0
        c1 = phi1 / (1 - phi2)
        ys = 1 / d * ((lamb1 * np.power(lamb2, xs) - lamb2 * np.power(lamb1, xs))
                      * c0 + (np.power(lamb1, xs) - np.power(lamb2, xs)) * c1)
        plt.errorbar(xs * scale, ys, fmt="-", label=f"tau = {tau*scale:.3f}")

        if log_plot:
            plt.semilogy()
        plt.grid()
        plt.legend()

    return tau * scale
