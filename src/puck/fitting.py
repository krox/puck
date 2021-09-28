""" fitting using minuit """

import iminuit
import numpy as np
import matplotlib.pyplot as plt
from puck.linear_fitting import *
from puck.statistics import *
import inspect
from uncertainties import ufloat

def fit_minuit(*args, model, guesses, xs=None, x_min=-float('inf'), x_max=float('inf'), plot=False, label=None, bootstrap=True, verbose=False):
    """
    fit 'model(x)' for arbitrary functions using minuit
    """

    # check and normalize data
    xs,ys,es,data = normalize_fit_input(*args)
    cond = (x_min <= xs) & (xs < x_max)

    # fit mean
    def f(*args):
        return np.sum(((ys[cond] - model(xs[cond], *args))/es[cond])**2)
    param_names = list(inspect.signature(model).parameters)[1:]
    m = iminuit.Minuit(f, *guesses, name=param_names)
    m.errordef = 1
    m.migrad()
    result_mean = list(m.values)
    chi2dof = f(*result_mean) / (xs[cond].shape[0] - len(param_names))

    # fit bootstraps
    if bootstrap is True:
        bootstrap = 500
    ys_bs = bootstrap_mean(data, samples=bootstrap)
    result_list = []
    for i in range(bootstrap):
        def f(*args):
            return np.sum(((ys_bs[i, cond] - model(xs[cond], *args))/es[cond])**2)
        m = iminuit.Minuit(f, *result_mean, name=param_names)
        m.errordef = 1
        m.migrad()
        result_list.append(list(m.values))
    result_list = np.array(result_list)
    result_err = result_list.std(axis=0)

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
        xs_plot = np.linspace(np.max([np.min(xs), x_min]), np.min([np.max(xs), x_max]))
        ys_plot = model(xs_plot, *result_mean)
        bs_plot = np.array([model(xs_plot, *result_list[i]) for i in range(bootstrap)])
        ys2_plot = bs_plot.mean(axis=0)
        es_plot = bs_plot.std(axis=0)

        # plot data + fit + error-band
        axs[0].errorbar(xs, ys, es, fmt="x", label=label)
        axs[0].errorbar(xs_plot, ys_plot, fmt="-", label=r"$\chi^2/dof={:.2f}$".format(chi2dof))
        axs[0].fill_between(xs_plot, ys2_plot - es_plot, ys2_plot + es_plot, alpha=0.5)
        axs[0].legend()
        axs[0].grid(True)

        # plot residuums + error-band
        if len(axs) >= 2:
            axs[1].errorbar(xs, ys - model(xs, *result_mean), es, fmt="x", label=r"$\chi^2/dof={:.2f}$".format(chi2dof))
            axs[1].errorbar(xs_plot, 0 * ys_plot, fmt="-")
            axs[1].fill_between(xs_plot, ys_plot - ys2_plot - es_plot, ys_plot - ys2_plot + es_plot, alpha=0.5)
            axs[1].grid(True)
            axs[1].legend()

    if verbose:
        print("chi2 / ({}-{}) = {:.2f}".format(xs[cond].shape[0], len(guesses), chi2dof))
        bias = (result_list.mean(axis=0) - result_mean)/result_mean
        for i in range(len(param_names)):
            print("{:8}: {:12}, error = {:5.2f}%, bias = {:5.2f}%".format(
                param_names[i],
                f"{ufloat(result_mean[i], result_err[i]):u2S}",
                np.abs(result_err[i]/result_mean[i])*100,
                np.abs(bias[i])*100
                ))

    return FitResult(values=result_mean, errors=result_err, param_names=param_names, chi2dof=chi2dof)


def fit_varpro(*args, models=None, guesses=None, x_min=-float('inf'), x_max=float('inf'), plot=False, label=None, bootstrap=True, verbose=False, param_names=None, plot_log=False):
    """
    fit 'c1 * model1(x,a) + c2 * model2(x,a) + ...' using 'projected variable method'
    returns [c1, c2, ..., a1, a2, ...]
    """

    # check and normalize data
    xs,ys,es,data = normalize_fit_input(*args)
    cond = (x_min <= xs) & (xs < x_max)
    xsc = xs[...,cond]
    ysc = ys[...,cond] # will be replaced during bootstrap
    esc = es[...,cond]

    if param_names is None:
        param_names = [f"c{i}" for i in range(len(models))] + [f"a{i}" for i in range(len(guesses))]
    assert len(param_names) == len(models) + len(guesses)

    # chi2 function to minimize
    def f(*a):
        ms = np.array([model(xsc, *a) for model in models]) # [models, chan, x]
        c = fit_linear_basic(ysc, esc, models=ms)
        return np.sum(((np.tensordot(c, ms, (0,0))- ysc)/esc)**2)

    # fit to the mean. if bootstrap is turned off:
    #    - use plotting capabilities of fit_linear()
    #    - error-bands, error-bars, and chi2 are not really correct, but might
    #      still be nice for fast testing
    minuit = iminuit.Minuit(f, *guesses)
    minuit.errordef = 1
    minuit.print_level = -1
    minuit.migrad()
    if not minuit.valid:
        raise RuntimeError(f"ERROR: mean-fit did not converge. (label={label})")
    chi2dof = minuit.fval / (ysc.size - len(guesses) - len(models))
    a_mean = list(minuit.values)
    a_err = list(minuit.errors)
    ms = [(lambda x, m=model: m(x, *a_mean)) for model in models]
    tmp = fit_linear(xs, ys, es, x_min=x_min, x_max=x_max, models=ms, plot=plot if not bootstrap else False, label=label)
    c_mean = np.array(tmp.values)
    c_err = np.array(tmp.errors)
    result_mean = np.append(c_mean, a_mean)
    if not bootstrap:
        return result_mean, np.array([*c_err, *a_err])

    # generate bootstrap samples
    if bootstrap is True:
        bootstrap = 500
    ys_bs = bootstrap_mean(data, samples=bootstrap)

    # repeat fit for bootstrap samples
    c_list = []
    a_list = []
    minuit_bs = iminuit.Minuit(f, *a_mean)
    minuit_bs.errors = a_err
    minuit_bs.errordef = 1
    minuit_bs.print_level = -1
    failed_bs = 0
    for i in range(ys_bs.shape[0]):
        ysc = ys_bs[i][..., cond]
        minuit_bs.reset()
        minuit_bs.migrad()
        if not minuit_bs.valid:
            failed_bs += 1
            continue
        a = list(minuit_bs.values)
        ms = np.array([model(xsc, *a) for model in models])
        c = fit_linear_basic(ysc, esc, models=ms)
        a_list.append(a)
        c_list.append(c)
    a_list = np.array(a_list)
    c_list = np.array(c_list)
    result_list = np.append(c_list, a_list, axis=1)
    result_err = result_list.std(axis=0, ddof=1)
    if failed_bs != 0:
        print(f"WARNING: {failed_bs} of {ys_bs.shape[0]} bootstrap samples did not converge. (label={label})")

    # and do our own plotting ( remember the difference between mean fit and mean of bootstrap-fits )
    if plot is True:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        if plot_log:
            axs[0].semilogy()
    elif type(plot) is np.ndarray or type(plot) is list:  # two plots for fit + residuals
        axs = plot
    elif plot:  # a single axes/figure/whatever object
        axs = np.array([plot, ])
    else:
        axs = None

    if axs is not None:
        def model_full(x, c, a):
            return sum(c[i] * models[i](x, *a) for i in range(len(models)))

        # determine fit function and error band
        xs_plot = np.linspace(np.max([np.min(xs), x_min]), np.min([np.max(xs), x_max]))
        ys_plot = model_full(xs_plot, c=c_mean, a=a_mean)
        bs_plot = np.array([model_full(xs_plot, c=c_list[k], a=a_list[k]) for k in range(ys_bs.shape[0])])
        ys2_plot = bs_plot.mean(axis=0)
        es_plot = bs_plot.std(axis=0)

        # plot everything (data + fit + error-band)

        if ys.ndim > 1:
            assert ys.ndim == 2
            if label is None:
                label = [None]*ys.shape[0]
            if type(label) != list:
                label = [label+f"[{i}]" for i in range(ys.shape[0])]
            for i in range(ys.shape[0]):
                axs[0].errorbar(xs, ys[i], es[i], fmt="x", label=label[i])
            for i in range(ys.shape[0]):
                axs[0].errorbar(xs_plot, ys_plot[i], fmt="-", label=r"$\chi^2/dof={:.2f}$".format(chi2dof) if i==0 else None)
            for i in range(ys.shape[0]):
                axs[0].fill_between(xs_plot, ys2_plot[i] - es_plot[i], ys2_plot[i] + es_plot[i], alpha=0.5)
        else:
            axs[0].errorbar(xs, ys, es, fmt="x", label=label)
            axs[0].errorbar(xs_plot, ys_plot, fmt="-", label=r"$\chi^2/dof={:.2f}$".format(chi2dof))
            axs[0].fill_between(xs_plot, ys2_plot - es_plot, ys2_plot + es_plot, alpha=0.5)
        axs[0].legend()
        axs[0].grid(True)

        if len(axs) >= 2:
            if ys.ndim > 1:
                for i in range(ys.shape[0]):
                    axs[1].errorbar(xs, ys[i] - model_full(xs, c, a)[i], es[i], fmt="x", label=label[i])
                axs[1].errorbar(xs_plot, 0.0, fmt="-", label=r"$\chi^2/dof={:.2f}$".format(chi2dof))
                for i in range(ys.shape[0]):
                    axs[1].fill_between(xs_plot, ys2_plot[i] - es_plot[i] - ys_plot[i], ys2_plot[i] + es_plot[i] - ys_plot[i], alpha=0.5)
            else:
                axs[1].errorbar(xs, ys - model_full(xs, c, a), es, fmt="x", label=label)
                axs[1].errorbar(xs_plot, 0.0, fmt="-", label=r"$\chi^2/dof={:.2f}$".format(chi2dof))
                axs[1].fill_between(xs_plot, ys2_plot - es_plot - ys_plot, ys2_plot + es_plot - ys_plot, alpha=0.5)
            axs[1].legend()
            axs[1].grid(True)

    if verbose:
        print("chi2 / ({}-{}) = {:.2f}".format(xsc.shape[0], len(guesses) + len(models), chi2dof))
        bias = (result_list.mean(axis=0) - result_mean)/result_mean
        for i in range(len(param_names)):
            print("{:8}: {:12}, error = {:5.2f}%, bias = {:5.2f}%".format(
                param_names[i],
                f"{ufloat(result_mean[i], result_err[i]):u2S}",
                np.abs(result_err[i]/result_mean[i])*100,
                np.abs(bias[i])*100
                ))

    return FitResult(values=result_mean, errors=result_err, param_names=param_names, chi2dof=chi2dof)


def fit_exp(*args, with_const=False, with_m=True, with_m2=False, with_degenerate=False, guess=None, **kwargs):
    """
    fit 'C0 + C1 * exp(-M1*x) + C2 * exp(-M2*x) + C3 * x * exp(-M1*x)'
       - fit is done using 'variable projected method'
       - errors are computed using bootstrap method
       - C0 and C2/M2 have to be turned on using with_const and with_m2
       - it should be 0 < M1 < M2
       - returns [C0, C1, C2, M1, M2] (or subsets thereof)
       - 'guess' should be a guess of M
    """
    if not with_m: assert with_degenerate

    if guess is None:
        guess = 0.5 # yeah, this is not very smart.

    models = []
    param_names = []
    if with_const:
        models.append(lambda x, m: np.ones_like(x))
        param_names.append("c0")
    if with_m:
        models.append(lambda x, m: np.exp(-m*x))
        param_names.append("c1")
    if with_degenerate:
        models.append(lambda x, m: x*np.exp(-m*x))
        param_names.append("c3")
    param_names.append("m")

    kwargs2 = kwargs.copy()
    if with_m2:
        kwargs2["verbose"] = False
        kwargs2["plot"] = False
    tmp = fit_varpro(*args, models=models, guesses=[guess], **kwargs2, param_names=param_names)
    mean = np.array(tmp.values)
    err = np.array(tmp.errors)

    if not with_m2:
        return tmp
    assert not with_degenerate # combined degenerate + m2 not implemented

    if with_const:
        models = [
            lambda x, m1, m2: np.ones_like(x),
            lambda x, m1, m2: np.exp(-m1*x),
            lambda x, m1, m2: np.exp(-m2*x)
        ]
        param_names = ["c0", "c1", "c2", "m1", "m2"]
    else:
        models = [
            lambda x, m1, m2: np.exp(-m1*x),
            lambda x, m1, m2: np.exp(-m2*x)
        ]
        param_names = ["c1", "c2", "m1", "m2"]

    return fit_varpro(*args, models=models, guesses=[0.8*mean[-1], 1.2*mean[-1]], **kwargs, param_names=param_names)

def fit_mass_multi(data, L, periodic=False, with_const=False, **kwargs):
    """ fit mass with multiple momenta simultaneously """
    # TODO: excited states, periodic boundaries
    assert data.ndim == 3
    mom2max = data.shape[1]-1
    T = data.shape[2]
    models = []
    for p2 in range(0, mom2max+1):
        e = np.zeros((mom2max+1, 1))
        e[p2] = 1.0
        if periodic:
            models.append(lambda x, m, e=e, p2=(p2*(2*np.pi/L)**2): (np.exp(-np.sqrt(m**2 + p2)*x)+np.exp(-np.sqrt(m**2 + p2)*(T-x)))*e)
        else:
            models.append(lambda x, m, e=e, p2=(p2*(2*np.pi/L)**2): np.exp(-np.sqrt(m**2 + p2)*x)*e)
    if with_const:
        models.append(lambda x, m, e=np.ones((mom2max+1,1)): x*0+e)

    if "label" not in kwargs:
        kwargs["label"] = [f"$p^2={p2}$" for p2 in range(0, mom2max+1)]
    if "param_names" not in kwargs:
        kwargs["param_names"] = [f"c{i}" for i in range(0, mom2max+1)]
        if with_const:
            kwargs["param_names"].append("A")
        kwargs["param_names"].append("m")


    return fit_varpro(data, models=models, guesses=[0.5], **kwargs)
