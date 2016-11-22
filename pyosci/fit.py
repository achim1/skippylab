"""
Provide routines for fitting charge histograms

"""

import numpy as np
from scipy.misc import factorial

from functools import reduce
from copy import deepcopy as copy
import scipy.optimize as optimize
import seaborn.apionly as sb
import dashi as d

from . import tools
from . import plotting as plt

from scipy.constants import elementary_charge as ELEMENTARY_CHARGE
d.visual()

import sys
import pylab as p


PALETTE = sb.color_palette("dark")
p.style.use("pyoscipresent")

def reject_outliers(data, m=2):
    """
    A simple way to remove extreme outliers from data

    Args:
        data (np.ndarray): data with outliers
        m (int): number of standard deviations outside the
                 data should be discarded

    Returns:
        np.ndarray
    """

    return data[abs(data - np.mean(data)) < m * np.std(data)]

def calculate_chi_square(data, model_data):
    """
    Very simple estimator for goodness-of-fit. Use with care.
    Non normalized bin counts are required.

    Args:
        data (np.ndarray): observed data (bincounts)
        model_data (np.ndarray): model predictions for each bin

    Returns:
        np.ndarray
    """

    chi = ((data - model_data)**2/data)
    return chi[np.isfinite(chi)].sum()


def poisson(lmbda, k):
    """
    Poisson distribution

    Args:
        lmbda (int): expected number of occurences
        k (int): measured number of occurences

    Returns:
        np.ndarrya
    """

    return np.power(lmbda, k) * np.exp(-1 * lmbda) / factorial(k)

def gauss(x, mu, sigma, n):
    """
    Returns a normed gaussian.

    Args:
        x (np.ndarray): x values
        mu (float): Gauss mu
        sigma (float): Gauss sigma
        n:

    Returns:

    """

    def _gaussnorm(sigma, n):
        return 1 / (sigma * np.sqrt(2 * n * np.pi))

    return _gaussnorm(sigma, n) * np.exp(-np.power((x - (n * mu)), 2) / (2 * n * (sigma ** 2)))

def calculate_sigma_from_amp(amp):
    """
    Get the sigma for the gauss from its peak value.
    Gauss is normed

    Args:
        amp (float):

    Returns:
        float
    """
    return 1/(np.sqrt(2*np.pi)*amp)


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

def exponential(x, N, n_noise):
    """
    An exponential model (for noise?)

    Args:
        x:
        N:
        n_noise:

    Returns:
        np.ndarray
    """
    return N * np.exp(-n_noise * x)

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

#func is of type func(xs, *params)

class Model(object):
    """
    Model holds theoretical prediction for
    data. Can be used to fit data.

    """

    def __init__(self, func, startparams):
        self._callbacks = [func]
        self.startparams = [*startparams]
        self.n_params = [len(startparams)]
        self.best_fit_params = [*startparams]
        self.coupling_variable = []
        self.all_coupled = False
        self.data = None
        self.xs = None
        self.chi2_ndf = None
        self.prediction = lambda xs: reduce(lambda x, y: x + y,\
                                  [f(xs) for f in self.components])


    def couple_models(self, coupling_variable):
        """
        Couple the models by a variable, which means use the variable not
        independently in all model components, but fit it only once.
        E.g. if there are 3 models with parameters p1, p2, k each and they
        are coupled by k, parameters p11, p21, p12, p22, and k will be fitted
        instead of p11, p12, k1, p21, p22, k2.

        Args:
            coupling_variable: variable number of the number in startparams

        Returns:
            None
        """
        assert len(np.unique(self.n_params)) == 1,\
            "Models have different numbers of parameters,difficult to couple!"

        self.coupling_variable.append(coupling_variable)

    def couple_all_models(self):
        """
        Use the first models startparams for
        the combined model

        Returns:
            None
        """
        self.all_coupled = True
        # if self.all_coupled:
        startparams = self.startparams[0:self.n_params[0]]

    def __add__(self, other):
        self._callbacks = self._callbacks + other._callbacks
        self.startparams = self.startparams + other.startparams
        self.n_params = self.n_params + other.n_params
        self.best_fit_params = self.startparams
        #self.best_fit_params = self.best_fit_params + other.best_fit_params
        return self

    @property
    def components(self):
        lastslice = 0
        thecomponents = []
        for i, cmp in enumerate(self._callbacks):
            thisslice = slice(lastslice,self.n_params[i] + lastslice)
            tmpcmp = copy(cmp)
            #thecomponents.append(lambda xs: tmpcmp(xs, ),*self.best_fit_params[thisslice])
            lastslice += self.n_params[i]
            best_fit = self.best_fit_params[thisslice]
            if self.all_coupled:
                best_fit = self.best_fit_params[0:self.n_params[0]]
            yield lambda xs: tmpcmp(xs, *best_fit)
        return thecomponents

    def __call__(self, xs, *params):
        """
        Give the model prediction


        Args:
            xs:

        Returns:

        """
        thecomponents = []
        firstparams = params[0:self.n_params[0]]
        first = self._callbacks[0](xs, *firstparams)

        lastslice = self.n_params[0]
        for i, cmp in enumerate(self._callbacks[1:]):
            thisslice = slice(lastslice, self.n_params[1:][i] + lastslice)
            # tmpcmp = copy(cmp)
            theparams = list(params[thisslice])
            if self.coupling_variable:
                for k in self.coupling_variable:
                    theparams[k] = firstparams[k]
            elif self.all_coupled:
                theparams = firstparams
            # thecomponents.append(lambda xs: tmpcmp(xs, *params[thisslice]))
            lastslice += self.n_params[1:][i]
            first += cmp(xs, *theparams)
        return first

    def add_data(self, data, nbins, subtract = None):
        """
        Add some data to the model, in preparation for the fit


        Args:
            data:
            nbins:
            subtract:

        Returns:

        """
        bins = np.linspace(min(data), max(data), nbins)
        self.data = d.factory.hist1d(data, bins).normalized(density=True)
        self.xs = self.data.bincenters

    def fit_to_data(self, data, nbins, subtract=None, **kwargs):
        """
        Apply this model to data

        Args:
            data (np.ndarray): the data, unbinned
            nbins (int): number of bins to put the data in
            subtract (func): subtract this from the data before fitting
            **kwargs: will be passed on to scipy.optimize.curvefit

        Returns:
            None
        """
        def model(xs, *params):
            thecomponents = []
            firstparams = params[0:self.n_params[0]]
            first = self._callbacks[0](xs, *firstparams)

            lastslice = self.n_params[0]
            for i, cmp in enumerate(self._callbacks[1:]):
                thisslice = slice(lastslice, self.n_params[1:][i] + lastslice)
                #tmpcmp = copy(cmp)
                theparams = list(params[thisslice])
                if self.coupling_variable:
                    for k in self.coupling_variable:
                        theparams[k] = firstparams[k]
                elif self.all_coupled:
                    theparams = firstparams
                #thecomponents.append(lambda xs: tmpcmp(xs, *params[thisslice]))
                lastslice += self.n_params[1:][i]
                first += cmp(xs, *theparams)
            return first

        startparams = self.startparams
        #if self.all_coupled:
        #    startparams = self.startparams[0:self.n_params[0]]
        bins = np.linspace(min(data), max(data), nbins)
        self.data = d.factory.hist1d(data, bins).normalized(density=True)
        self.xs = self.data.bincenters

        h = d.factory.hist1d(data, bins)
        h_norm = h.normalized(density=True)
        xs = h_norm.bincenters
        data = self.data.bincontent
        if subtract is not None:
            data -= subtract(self.xs)

        print("Using start params...", startparams)

        fitkwargs = {"maxfev": 1000000, "xtol": 1e-10, "ftol": 1e-10}
        if "bounds" in kwargs:
            fitkwargs.pop("maxfev")
            fitkwargs["max_nfev"] = 1000000
        fitkwargs.update(kwargs)
        parameters, covariance_matrix = optimize.curve_fit(model, self.xs, \
                                                           data, p0=startparams, \
                                                           # bounds=(np.array([0, 0, 0, 0, 0] + [0]*len(start_params[5:])),\
                                                           # np.array([np.inf, np.inf, np.inf, np.inf, np.inf] +\
                                                           # [np.inf]*len(start_params[5:]))),\
                                                           # max_nfev=100000)
                                                           # method="lm",\
                                                           **fitkwargs)

        print("Fit yielded parameters", parameters)
        print("Covariance matrix", covariance_matrix)
        print("##########################################")

        # simple GOF
        norm = h.bincontent / h_norm.bincontent
        norm = norm[np.isfinite(norm)][0]
        chi2 = (calculate_chi_square(h.bincontent, norm * model(h.bincenters, *parameters)))
        self.chi2_ndf = chi2/nbins
        print("Obtained chi2 and chi2/ndf of {:4.2f} {:4.2f}".format(chi2, chi2 / nbins))
        self.best_fit_params = parameters
        return parameters
        #self.best_fit_params = fit_model(data, nbins, model, startparams, **kwargs)

    def clear(self):
        """
        Reset the model

        Returns:
            None
        """
        self.__init__(self._callbacks[0], self.startparams[:self.n_params])

    def plot_result(self, ymin=1000, ylabel="normed bincount", xlabel="Q [C]", fig=None,\
                    add_parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\",0))):
        """
        Show the fit result

        Args:
            ymin (float): limit the yrange
            fig (pylab.figure): A figure instance
            add_parameter_text (tuple): Display a parameter in the table on the plot
                                        ((text, parameter_number), (text, parameter_number),...)
        Returns:
            pylab.figure
        """
        if fig is None:
            fig = p.figure()
        ax = fig.gca()
        self.data.scatter(color="k")
        ax.plot(self.xs, self.prediction(self.xs), color=PALETTE[2])
        for comp in self.components:
            ax.plot(self.xs, comp(self.xs), linestyle=":", lw=1, color="k")

        ax.set_yscale("log")
        ax.set_ylim(ymin=ymin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        infotext = r"\begin{tabular}{ll}"
        infotext += r"$\chi^2/ndf$ & {:4.2f}\\".format(self.chi2_ndf)
        infotext += r"entries& {}\\".format(self.data.stats.nentries)
        if add_parameter_text is not None:
            for partext in add_parameter_text:
                print (partext[0])
                print (partext[1])
                infotext += partext[0].format(self.best_fit_params[partext[1]])
            #infotext += r"$\mu_{{SPE}}$& {:4.2e}\\".format(self.best_fit_params[mu_spe_is_par])
        infotext += r"\end{tabular}"
        ax.text(0.9, 0.9, infotext,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
        sb.despine()
        return fig

#mu_p, sigma_p, mu, sigma , lmbda
def fit_model_deprecated(data, nbins, model, start_params, **kwargs):
    """


    Args:
        data (np.ndarray):
        nbins (int)
        model (func):
        start_params (tuple):
        **kwargs: will be passed on to scipy.optimize.curve_fit

    Keyword Args:
        xerr (np.ndarray):

    Returns:
        tuple
    """

    bins = np.linspace(min(data), max(data), nbins)
    h = d.factory.hist1d(data,bins)
    h_norm = h.normalized(density=True)
    xs = h_norm.bincenters
    data = h_norm.bincontent
    #mu_p, sigma_p, mu, sigma , lmbda
    print ("Using start params...", start_params)

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

    # simple GOF
    norm = h.bincontent / h_norm.bincontent
    norm = norm[np.isfinite(norm)][0]
    chi2 = (calculate_chi_square(h.bincontent, norm * model(h.bincenters, *parameters)))
    print("Obtained chi2 and chi2/ndf of {:4.2f} {:4.2f}".format(chi2, chi2 / nbins))
    return parameters


def fit_model(waveformfile, model, startparams, \
              rej_outliers=True, nbins=200, \
              parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\", 5),), **kwargs):
    """
    Standardazied fitting routine

    Args:
        waveformfile (str): full path to a file with waveforms saved by pyosci
        model (pyosci.fit.Model): A model to fit to the data
        startparams (tuple): initial parameters to model

    Keyword Args:
        rej_outliers (bool): Remove extreme outliers from data
        nbins (int): Number of bins
        parameter_text (tuple): will be passed to model.plot_result
    Returns:
        tuple
    """
    head, wf = tools.load_waveform(waveformfile)
    plt.plot_waveform(head, tools.average_wf(wf))
    charges = 1e12 * np.array([-1 * tools.integrate_wf(head, w) for w in wf])
    # charges += np.ones(len(charges))
    if rej_outliers:
        charges = reject_outliers(charges)
    model.startparams = startparams
    model.fit_to_data(charges, nbins, **kwargs)
    fig = model.plot_result(ymin=1e-4, \
                            add_parameter_text=parameter_text, \
                            xlabel=r"$Q$ [pC]")
    if hasattr(model, "parameter_names"):
        pretty_pars = [k for k in zip(model.parameter_names, model.best_fit_params)]
    print("Best fit parameters {}".format(model.best_fit_params))
    ax = fig.gca()
    ax.grid(1)
    return ax, model

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
