""" general linear fitting (Direct matrix inversion. No Minuit) """

import numpy as np


def fit_linear_basic(ys, es, models):
    models = np.array(models)
    assert ys.ndim>=1 and es.ndim>=1 and models.ndim>=2
    assert ys.shape==es.shape==models.shape[1:], f"incompatible shapes ys={ys.shape}, es={es.shape}, models={models.shape}"
    ys = np.ravel(ys)
    es = np.ravel(es)
    models = np.reshape(models, (models.shape[0], -1))
    A = np.transpose(models) / es[:, np.newaxis]
    B = np.linalg.pinv(A)
    r = ys / es
    return B@r


def normalize_fit_input(first, second=None, third=None):
    """
    Data to fitting algorithms can be provided in various forms:
    fit(xs, ys, es)
    fit(xs, data)
    fit(ys, es)
    fit(data)

    where:
      * xs defaults to np.arange(ys.shape[-1])
      * the first dimension of 'data' is assumed to contain independent samples
      * es can be an array, or a single number

    This function takes any of these signatures and returns (xs,ys,es,data)
      * data=None if not provided
      * shapes are all checked
      * everything is converted to numpy-arrays
    """
    first = np.array(first)
    second = np.array(second) if second is not None else None
    third = np.array(third) if third is not None else None

    if first.ndim>=2 and second is None and third is None: # (data)
        xs = np.arange(first.shape[-1])
        ys = first.mean(axis=0)
        es = first.std(axis=0, ddof=1)/first.shape[0]**0.5
        data = first
    elif first.ndim>=1 and (second.ndim==first.ndim or second.ndim==0) and third is None: # (ys, es)
        xs = np.arange(first.shape[-1])
        ys = first
        es = second
        data = None
    elif first.ndim==1 and second.ndim>=2 and third is None: # (xs, data)
        xs = first
        ys = second.mean(axis=0)
        es = second.std(axis=0, ddof=1)/second.shape[0]**0.5
        data = second
    elif first.ndim==1 and second.ndim>=1 and (third.ndim==second.ndim or third.ndim==0): # (xs, ys, es)
        xs = first
        ys = second
        es = third
        data = None
    else:
        assert False

    es = np.broadcast_to(es, shape=ys.shape)

    assert xs.ndim==1 and xs.shape[0]==ys.shape[-1]
    assert ys.shape==es.shape
    if data is not None:
        assert data.shape[-1]==xs.shape[0]
    return xs, ys, es, data

def fit_linear(*args, models=None, plot=False, with_zero=False, label=None, verbose=False, param_names=None, x_min=-float('inf'), x_max=float('inf')):
    """
    fit 1D data as linear combination of models.
      - data is assumed to be uncorrelated gaussian
      - 'models' should be a list of lambdas
      - 'with_zero' and 'label' are just for plotting
      - returns fitted parameters and their errors
    """

    # normalize and check data input
    xs,ys,es,data = normalize_fit_input(*args)

    # determine prefactors from simple linear optimization
    cond = (x_min <= xs) & (xs < x_max)
    A = np.transpose(np.array([m(xs[...,cond]) for m in models]).reshape((len(models),-1))) / es[...,cond].ravel()[:, np.newaxis]
    B = np.linalg.pinv(A)
    r = ys[...,cond].ravel() / es[...,cond].ravel()
    a = B @ r
    cov = B @ np.transpose(B)
    a_err = np.sqrt(np.diag(cov))

    # simple chi2 test as quality-of-fit indicator
    chi2 = ((A@a - r)**2).sum()
    dof = A.shape[0] - len(models)

    if ys.ndim > 1:
        assert plot is False, "plotting of multiple channels not supported (yet)"

    if plot is True:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    elif type(plot) is np.ndarray or type(plot) is list:  # two plots for fit + residuals
        axs = plot
    elif plot:  # a single axes/figure/whatever object
        axs = np.array([plot, ])
    else:
        axs = None

    if axs is not None:
        # determine fit function and error band
        if with_zero:
            assert x_min == -float('inf'), "cant use 'with_zero' and 'x_min' at the same time"
            xs_plot = np.linspace(np.min([0, *xs]), np.min([np.max([0, *xs]), x_max]))
        else:
            xs_plot = np.linspace(np.max([np.min(xs), x_min]), np.min([np.max(xs), x_max]))
        ys_plot = (np.array([m(xs_plot) for m in models])
                   * a[:, np.newaxis]).sum(axis=0)
        M = np.transpose(np.array([m(xs_plot) for m in models]))
        es_plot = np.sqrt(np.einsum("ij,jk,ik->i", M, cov, M))

        # first plot: data + fit + error-band
        axs[0].errorbar(xs, ys, es, fmt="x", label=label)
        axs[0].errorbar(xs_plot, ys_plot, fmt="-", label=r"$\chi^2/\mathrm{{dof}} = {:.2f}$".format(chi2/dof))
        axs[0].fill_between(xs_plot, ys_plot - es_plot,
                            ys_plot + es_plot, alpha=0.5)
        axs[0].legend()
        axs[0].grid(True)

        # second plot: residuals
        if len(axs) >= 2:
            axs[1].errorbar(xs, ys - np.transpose(np.array([m(xs) for m in models])) @ a, es, fmt="x", label=r"$\chi^2/\mathrm{{dof}} = {:.2f}$".format(chi2/dof))
            axs[1].errorbar(xs_plot, 0 * ys_plot, fmt="-")
            axs[1].fill_between(xs_plot, ys_plot - M@a - es_plot, ys_plot - M@a + es_plot, alpha=0.5)
            axs[1].grid(True)
            axs[1].legend()

    if verbose:
        from uncertainties import ufloat
        if param_names is None:
            param_names = [f"a_{i}" for i in range(len(a))]
        print("chi^2/dof = {:.3f}".format(chi2 / dof))
        for i in range(len(a)):
            print("{} = {:u2S} ({:.2f} %)".format(
                param_names[i], ufloat(a[i], a_err[i]), a_err[i] / np.abs(a[i]) * 100))

    return a, a_err


def fit_poly(*args, exponents=[0, 1, 2], offset=0.0, **kwargs):
    """
    use 'fit_linear' to fit some polynomials
    """
    models = []
    param_names = [f"a{i}" for i in exponents]
    models = [(lambda e: lambda x: (x-offset)**e)(i) for i in exponents]
    return fit_linear(*args, models=models, **kwargs, param_names=param_names)
