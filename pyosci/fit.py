"""
Provide routines for fitting charge histograms

"""

import numpy as np
from scipy.misc import factorial
from functools import reduce
import scipy.optimize as optimize
import seaborn.apionly as sb
import dashi as d
from scipy.constants import elementary_charge as ELEMENTARY_CHARGE
d.visual()

import sys
import pylab as p


PALETTE = sb.color_palette("dark")
p.style.use("pyoscipresent")


def _gaussnorm(sigma, n):
    return 1 / (sigma * np.sqrt(2 * n * np.pi))


def poisson(x, lmbda, k):
    #k = int(k)
    return np.power(lmbda, k) * np.exp(-1 * lmbda) / factorial(k)


def gauss(x, mu, sigma, n):
    return _gaussnorm(sigma, n) * np.exp(-np.power((x - (n * mu)), 2) / (2 * n * (sigma ** 2)))

def charge_response_model(x,*params,return_ped=False):
    """
    A combination of gaussian weighted by the poisson probability to measure zero

    Args:
        x:
        params (tuple):  mu_p, sigma_p, mu, sigma, lmbda, w1...wn

    Keyword Args:
        n_max (int): max number of gaussians to sum up

    Returns:
        np.ndarray
    """
    #print (params)
    n_max = len(params[5:])
    mu_p = params[0]
    sigma_p = params[1]
    mu = params[2]
    sigma = params[3]
    lmbda = params[4]
    weights = params[5:]
    #weights = np.ones(len(params[5:]))
    #print weights
    firstmember = weights[0]*poisson(x,lmbda,0)*gauss(x, mu_p, sigma_p, 1)
    rest = [weights[n-1]*poisson(x, mu, n)*gauss(x, mu, sigma, n) for n in range(1, len(params[5:]))]
    try:
        result = reduce(lambda x, y: x+y, rest) + firstmember
    except TypeError:
        result = weights[0]*poisson(x, mu, 1)*gauss(x, mu, sigma, 1) + firstmember
    if return_ped:
        return firstmember, rest
    return result

def charge_response_with_noise(x, amp_p, mu_p, sigma_p, amp, mu, sigma, N, n_noise,\
                               return_components=False):
    """
    A different charge response model attributing an exponential dropping noise term

    Args:
        x:
        mu_p:
        sigma_p:
        mu:
        sigma:
        N:
        n_noise:

    Keyword Args:
        return_components (bool):

    Returns:

    """
    pedestal = gauss(x, mu_p, sigma_p,1)
    sp = amp*gauss(x, mu, sigma,1)
    noise = N*np.exp(-n_noise*x)
    if return_components:
        return pedestal, sp, noise*np.ones(len(x))
    return pedestal + sp + noise

def error_func(f, x, y_true, parameters):
    """
    Get the errorfunction for a certain model

    Args:
        f (func): the model
        x (np.ndarray): x values to evaluete f
        parameters (iterable): paramters for f
        y_true (np.ndarray): true data

    Returns:
        func
    """
    return f(x, *parameters) - y_true

#mu_p, sigma_p, mu, sigma , lmbda


def fit_model(xs, data, model, start_params,  **kwargs):
    """


    Args:
        histogram:
        model:
        start_params:
        **kwargs: will be passed on to scipy.optimize.curve_fit

    Keyword Args:
        xerr (np.ndarray):

    Returns:

    """

    #mu_p, sigma_p, mu, sigma , lmbda
    xerr = None
    if "xerr" in kwargs:
        xerr = kwargs.pop("xerr")

    print ("Using start params...", start_params)
    #if fitrange is not None:
    #    print ("Constraining fit to {}".format(fitrange))
    #    fitrange = np.asarray(fitrange)
    #    xs = xs[np.logical_and(xs >= fitrange[0], xs <= fitrange[1])]
    #    data = data[np.logical_and(xs >= fitrange[0], xs <= fitrange[1])]

    fitkwargs = {"maxfev" : 1000000, "xtol": 1e-10, "ftol": 1e-10}
    if "bounds" in kwargs:
        fitkwargs.pop("maxfev")
        fitkwargs["max_nfev"] = 1000000
    fitkwargs.update(kwargs)
    parameters, covariance_matrix = optimize.curve_fit(model,xs,\
                                                       data, p0=start_params,\
                                                       # bounds=(np.array([0, 0, 0, 0, 0] + [0]*len(start_params[5:])),\
                                                       # np.array([np.inf, np.inf, np.inf, np.inf, np.inf] +\
                                                       # [np.inf]*len(start_params[5:]))),\
                                                       # max_nfev=100000)
                                                       # method="lm",\
                                                       **fitkwargs)


    print ("Fit yielded parameters", parameters)
    print ("Covariance matrix" , covariance_matrix)
    print ("##########################################")

    chisquare = 0.
    deviations = error_func(model, xs, data, parameters)

    for i, d in enumerate(deviations):
        chisquare += d * d / model(xs[i],*parameters)
    if xerr is not None:
        chi_squared = np.sum(((model(xs, *parameters) - data) / xerr) ** 2)
        reduced_chi_squared = (chi_squared) / (len(xs) - len(parameters))
        print ("Obtanied chisquared/ndf of {:4.2f}".format(reduced_chi_squared))

    chisquare_ndf = chisquare/(len(xs) - len(parameters))
    print("Obtained chisquare of {:4.2f}".format(chisquare))
    print("Obtained chisquare/ndf of {:4.2f}".format(chisquare_ndf))
    return parameters

if __name__ == "__main__":

    charge_factor = 1/ELEMENTARY_CHARGE
    conversion = charge_factor/1e10
    conversion = 1.
    charge_factor = 1e10
    print (charge_factor)
    print (conversion)
    nbins = 70
    charges = abs(np.load(sys.argv[1])) * charge_factor #1e10
    print(charges)
    bin_edges = np.linspace(min(charges), max(charges), nbins)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    hist, edges = np.histogram(charges, bin_edges, normed=False)
    histvals, __ = np.histogram(charges, bin_edges, density=True, normed=True)

    # mu_p, sigma_p, mu, sigma , lmbda
    start_params = np.array([0, 10, 1, .5, 0.] + [1] * 10)  # ,1,1,1])
    # mu_p, sigma_p, mu, sigma , N, n_noise
    start_params_noise = np.array([10, 0, 1*conversion, .1, 2*conversion , .5*conversion , 1., 1*conversion ])
    #start_params_noise = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    #params_uno = fit_model(bin_centers, histvals, charge_response_model, start_params)
    params_duo = fit_model(bin_centers, histvals, charge_response_with_noise, start_params_noise)
    #pred_uno = charge_response_model(bin_centers, *params_uno)
    pred_duo = charge_response_with_noise(bin_centers, *params_duo)

    #pedestal_uno, contribs_uno = charge_response_model(bin_centers, *params_uno, return_ped=True)

    # for i,name in zip(cpars,["mu_p","sigma_p", "mu", "sigma", "lambda"] + ["w"]*len(cpars[5:])):
    #    print (i,name)
    # mu_p, sigma_p, mu, sigma , lmbda

    print (params_duo[4])
    mu_peak = (1e-10)*params_duo[4]*50*(1e-12) # units from integration mV, ns, and 50 ohms impedance and the 1e-10 we had in the beginning

    print (ELEMENTARY_CHARGE)
    gain = mu_peak/ELEMENTARY_CHARGE
    print (mu_peak)
    print (gain)
    fig = p.figure()
    ax = fig.gca()
    # evaluate contributions
    for contrib in charge_response_with_noise(bin_centers, *params_duo, return_components=True):
        ax.plot(bin_centers, contrib, linestyle=":", lw=1, color="k")

    #ax.plot(bin_centers, pedestal_uno, linestyle=":", lw=1, color="k")
    #
    #for i, contrib in enumerate(contribs_uno):
    #    ax.plot(bin_centers, contrib, linestyle=":", lw=1, color="k")

    # ax.plot(bin_centers, pred_uno, lw=2, color=PALETTE[0], label="Multi PE model")
    ax.plot(bin_centers, pred_duo, lw=2, color=PALETTE[1], label="Noise model")
    h = d.factory.hist1d(charges,nbins)
    #h.bincontent = h.bincontent/h.bincontent.sum()
    #h.binerror = h.binerror/h.bincontent.sum()
    h = h.normalized(density=True)
    h.scatter(color="k", linewidth= 3, alpha= 1.)
    #ax.errorbar(bin_centers, histvals)
    ax.set_yscale("log")

    x_right_edge = max(h.bincenters[h.bincontent > 0])
    y_highest = max(h.bincontent)*10
    ax.set_xlim(xmax=x_right_edge,xmin=0)
    ax.grid(True)
    ax.set_ylim(ymin=1e-5,ymax=y_highest)
    sb.despine()
    ax.set_xlabel("Q [a. u.]")
    ax.set_ylabel("pdf")
    ax.text(8,1,"$\mu_{SPE}$ %4.2f" %params_duo[4])
    fig.savefig(sys.argv[2],bbox_inches="tight")
    p.show()
