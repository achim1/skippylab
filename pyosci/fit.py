"""
Provide routines for fitting charge histograms
"""

import sys
import pylab as p
import numpy as np
from scipy.misc import factorial

from functools import reduce
from copy import deepcopy as copy
from collections import namedtuple

import scipy.optimize as optimize
import seaborn.apionly as sb
import dashi as d

from . import tools
from . import plotting as plt


from scipy.constants import elementary_charge as ELEMENTARY_CHARGE
d.visual()

try:
    from iminuit import Minuit
except ImportError:
    print ("WARNING, can not load iminuit")

# default color palette
PALETTE = sb.color_palette("dark")
p.style.use("pyoscipresent")


def get_n_hit(charges, nbins):
    """
    Identify how many events are in the pedestal

    Args:
        charges (np.ndarray): The measured charges
        nbins (int): number of bins to use

    Returns:
        tuple (n_hit, n_all)
    """
    one_gauss = lambda x, n, y, z: n * gauss(x, y, z, 1)
    ped_mod = Model(one_gauss, (1000, -.1, 1))
    ped_mod.fit_to_data(charges, nbins, normalize=False, silent=False)
    n_hit = abs(ped_mod.data.bincontent - ped_mod.prediction(ped_mod.xs)).sum()
    n_pedestal = ped_mod.data.stats.nentries - n_hit
    n_all = ped_mod.data.stats.nentries
    return n_hit, n_all


def calculate_mu(charges, nbins):
    """
    Calculate mu out of
    P(hit) = (N_hit/N_all) = exp(QExCExLY)
    where P is the probability for a hit, QE is quantum efficiency, CE is
    """
    n_hit, n_all = get_n_hit(charges, nbins)
    n_pedestal = n_all - n_hit
    #mu = -1 * np.log(n_pedestal / n_all)
    mu = -1 * np.log(1 - (n_hit/n_all))
    return mu


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


def calculate_gain(mu_ped, mu_spe, prefactor=1e-12):
    """
    Calculate the pmt gain from the charge distribution

    Args:
        mu_ped (float): the mean of the gaussian fitting the pedestal
        mu_spe (float): the mean of the gaussian fitting the spe response

    Keyword Args:
        prefactor (float): unit of charge (default pico coulomb)

    Returns:
        float
    """
    charge =  abs(mu_spe) - abs(mu_ped)
    charge *= prefactor
    return charge/ELEMENTARY_CHARGE


def chi2_exponential_part(xs, data, pred, mu_ped, mu_ser):
    mask = np.logical_and(xs >= mu_ped, xs <= mu_ser)
    data = data[mask]
    pred = pred[mask]
    xs   = xs[mask]
    chi2_ndf = calculate_chi_square(data, pred)/len(xs)
    return chi2_ndf


def calculate_peak_to_valley_ratio(bestfitmodel, mu_ped, mu_spe, control_plot=False):
    """
    Calculate the peak to valley ratio
    Args:
        bestfitmodel (fit.Model): A fitted model to charge response data
        mu_ped (float): The x value of the fitted pedestal
        mu_spe (flota): The x value of the fitted spe peak
   
    Keyword Args:
        control_plot (bool): Show control plot to see if correct values are found
   
    """
    

    tmpdata = bestfitmodel.prediction(bestfitmodel.xs)
    valley = min(tmpdata[np.logical_and(bestfitmodel.xs > mu_ped,\
                                    bestfitmodel.xs < mu_spe)])
    valley_x = bestfitmodel.xs[tmpdata == valley]

    peak = max(tmpdata[bestfitmodel.xs > valley_x])
    peak_x = bestfitmodel.xs[tmpdata == peak]
    peak_v_ratio = (peak/valley)
    
    if control_plot:
        fig = p.figure()
        ax = fig.gca()
        ax.plot(bestfitmodel.xs,tmpdata)
        print (valley)
        print (valley_x)

        ax.scatter(valley_x,valley,marker="o")
        ax.scatter(peak_x, peak, marker="o")
        ax.set_ylim(ymin=1e-4)
        ax.set_yscale("log")
        ax.grid(1)
        sb.despine()

    return peak_v_ratio


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


class Model(object):
    """
    Model holds theoretical prediction for
    data. Can be used to fit data.

    """

    def __init__(self, func, startparams=None, norm=1):
        """
        Initialize a new model


        Args:
            func: the function to predict the data
            startparams (tuple): A set of startparameters.
            norm: multiply the result of func when evaluated with norm
        """

        # if no startparams are given, construct 
        # and initialize with 0es.
        # FIXME: better estimate?
        if startparams is None:
            startparams = [0]*func.__code__.co_argcount

        def normed_func(*args, **kwargs):
            return norm*func(*args, **kwargs)

        self._callbacks = [normed_func]
        self.startparams = [*startparams]
        self.n_params = [len(startparams)]
        self.best_fit_params = [*startparams]
        self.coupling_variable = []
        self.all_coupled = False
        self.data = None
        self.xs = None
        self.chi2_ndf = None
        self.chi2_ndf_components = []
        self.norm = None
        self.prediction = lambda xs: reduce(lambda x, y: x + y,\
                                  [f(xs) for f in self.components])
        self.first_guess = None

    def add_first_guess(self, func):
        """
        Use func to estimate better startparameters

        Args:
            func: Has to yield a set of startparameters

        Returns:

        """

        assert self.all_coupled, "Does not work yet if not all variables are coupled"
        assert func.__code__.co_argcount == len(self.startparams), "first guess algorithm must yield startparams!"
        self.first_guess = func

    def eval_first_guess(self, data):
        self.startparams = self.first_guess(data)

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
        #return thecomponents

    def __call__(self, xs, *params):
        """
        Give the model prediction

        Args:
            xs (np.ndaarray): the values the model should be evaluated on

        Returns:
            np.ndarray
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

    def fit_to_data(self, data, nbins, silent=False, subtract=None,\
                    normalize=True, **kwargs):
        """
        Apply this model to data

        Args:
            data (np.ndarray): the data, unbinned
            nbins (int): number of bins to put the data in
            silent (bool): silence output
            subtract (func): subtract this from the data before fitting
            normalize (bool): normalize data before fitting
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
        self.data = d.factory.hist1d(data, bins)
        if normalize:
            self.data = self.data.normalized(density=True)
        self.xs = self.data.bincenters

        h = d.factory.hist1d(data, bins)
        h_norm = h.normalized(density=True)
        xs = h_norm.bincenters
        data = self.data.bincontent
        if subtract is not None:
            data -= subtract(self.xs)

        if not silent: print("Using start params...", startparams)

        fitkwargs = {"maxfev": 1000000, "xtol": 1e-10, "ftol": 1e-10}
        if "bounds" in kwargs:
            fitkwargs.pop("maxfev")
            # this is a matplotlib quirk
            fitkwargs["max_nfev"] = 1000000
        fitkwargs.update(kwargs)
        parameters, covariance_matrix = optimize.curve_fit(model, self.xs,\
                                                           data, p0=startparams,\
                                                           # bounds=(np.array([0, 0, 0, 0, 0] + [0]*len(start_params[5:])),\
                                                           # np.array([np.inf, np.inf, np.inf, np.inf, np.inf] +\
                                                           # [np.inf]*len(start_params[5:]))),\
                                                           # max_nfev=100000)
                                                           # method="lm",\
                                                           **fitkwargs)

        if not silent: print("Fit yielded parameters", parameters)
        if not silent: print("{:4.2f} NANs in covariance matrix".format(len(covariance_matrix[np.isnan(covariance_matrix)])))
        if not silent: print("##########################################")

        # simple GOF
        norm = 1
        if normalize:
            norm = h.bincontent / h_norm.bincontent
            norm = norm[np.isfinite(norm)][0]

        self.norm = norm
        chi2 = (calculate_chi_square(h.bincontent, norm * model(h.bincenters, *parameters)))
        self.chi2_ndf = chi2/nbins

        # FIXME: new feature
        #for cmp in self.components:
        #    thischi2 = (calculate_chi_square(h.bincontent, norm * cmp(h.bincenters)))
        #    self.chi2_ndf_components.append(thischi2/nbins)

        if not silent: print("Obtained chi2 and chi2/ndf of {:4.2f} {:4.2f}".format(chi2, chi2 / nbins))
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

    def plot_result(self, ymin=1000,xmax=8, ylabel="normed bincount",\
                    xlabel="Q [C]", fig=None,\
                    log=True,\
                    model_alpha=.3,\
                    add_parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\",0),)):
        """
        Show the fit result

        Args:
            ymin (float): limit the yrange to ymin
            xmax (float): limit the xrange to xmax
            model_alpha (float): 0 <= x <= 1 the alpha value of the lineplot
                                for the model
            ylabel (str): label for yaxis
            log (bool): plot in log scale
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
        ax.plot(self.xs, self.prediction(self.xs), color=PALETTE[2], alpha=model_alpha)
        for comp in self.components:
            ax.plot(self.xs, comp(self.xs), linestyle=":", lw=1, color="k")

        if log: ax.set_yscale("log")
        ax.set_ylim(ymin=ymin)
        ax.set_xlim(xmax=xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        infotext = r"\begin{tabular}{ll}"
        infotext += r"$\chi^2/ndf$ & {:4.2f}\\".format(self.chi2_ndf)
        infotext += r"entries& {}\\".format(self.data.stats.nentries)
        if add_parameter_text is not None:
            for partext in add_parameter_text:
                infotext += partext[0].format(self.best_fit_params[partext[1]])
            #infotext += r"$\mu_{{SPE}}$& {:4.2e}\\".format(self.best_fit_params[mu_spe_is_par])
        infotext += r"\end{tabular}"
        ax.text(0.9, 0.9, infotext,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
        sb.despine()
        return fig


def pedestal_fit(filename, nbins, fig=None):
    """
    Fit a pedestal to measured waveform data
    One shot function for
    * integrating the charges
    * making a histogram
    * fitting a simple gaussian to the pedestal
    * calculating mu
        P(hit) = (N_hit/N_all) = exp(QExCExLY)
        where P is the probability for a hit, QE is quantum efficiency,
        CE is the collection efficiency and
        LY the (unknown) light yield

    Args:
        filename (str): Name of the file with waveform data
        nbins (int): number of bins for the underlaying charge histogram

    """

    head, wf = tools.load_waveform(filename)
    charges = -1e12 * tools.integrate_wf(head, wf)
    plt.plot_waveform(head, tools.average_wf(wf))
    p.savefig(filename.replace(".npy", ".wf.pdf"))
    one_gauss = lambda x, n, y, z: n * fit.gauss(x, y, z, 1)
    ped_mod = fit.Model(one_gauss, (1000, -.1, 1))
    ped_mod.fit_to_data(charges, nbins, normalize=False)
    fig = ped_mod.plot_result(add_parameter_text=((r"$\mu_{{ped}}$& {:4.2e}\\", 1), \
                                                  (r"$\sigma_{{ped}}$& {:4.2e}\\", 2)), \
                              xlabel=r"$Q$ [pC]", ymin=1, xmax=8, model_alpha=.2, fig=fig, ylabel="events")

    ax = fig.gca()
    n_hit = abs(ped_mod.data.bincontent - ped_mod.prediction(ped_mod.xs)).sum()
    ax.grid(1)
    bins = np.linspace(min(charges), max(charges), nbins)
    data = d.factory.hist1d(charges, bins)
    n_pedestal = ped_mod.data.stats.nentries - n_hit

    mu = -1 * np.log(n_pedestal / ped_mod.data.stats.nentries)

    print("==============")
    print("All waveforms: {:4.2f}".format(ped_mod.data.stats.nentries))
    print("HIt waveforms: {:4.2f}".format(n_hit))
    print("NoHit waveforms: {:4.2f}".format(n_pedestal))
    print("mu = -ln(N_PED/N_TRIG) = {:4.2e}".format(mu))

    ax.fill_between(ped_mod.xs, 1e-4, ped_mod.prediction(ped_mod.xs),\
                    facecolor=PALETTE[2], alpha=.2)
    p.savefig(filename.replace(".npy", ".pdf"))
    # xs = self.data.bincenters
    return ped_mod

################################################

def fit_model(charges, model, startparams=None, \
              rej_outliers=False, nbins=200, \
              silent=False,\
              parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\", 5),),
              use_minuit=False,\
              normalize=True,\
              **kwargs):
    """
    Standardazied fitting routine

    Args:
        charges (np.ndarray): Charges obtained in a measurement (no histogram)
        model (pyosci.fit.Model): A model to fit to the data
        startparams (tuple): initial parameters to model, or None for first guess

    Keyword Args:
        rej_outliers (bool): Remove extreme outliers from data
        nbins (int): Number of bins
        parameter_text (tuple): will be passed to model.plot_result
        use_miniuit (bool): use minuit to minimize startparams for best 
                            chi2
        normalize (bool): normalize data before fitting
        silent (bool): silence output
    Returns:
        tuple
    """
    if rej_outliers:
        charges = reject_outliers(charges)
    if use_minuit:

        from iminuit import Minuit

        # FIXME!! This is too ugly. Minuit wants named parameters ... >.<

        assert len(startparams) > 10; "Currently more than 10 paramters are not supported for minuit fitting!"
        assert model.all_coupled, "Minuit fitting can only be done for models with all parmaters coupled!"
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

        funcstring = "def do_min("
        for i,__ in enumerate(startparams):
            funcstring += names[i] + ","
        funcstring = funcstring[:-1] + "):\n"
        funcstring += "\tmodel.startparams = ("
        for i,__ in enumerate(startparams):
            funcstring += names[i] + ","
        funcstring = funcstring[:-1] + ")\n"
        funcstring += "\tmodel.fit_to_data(charges, nbins, silent=True, **kwargs)"
        funcstring += "\treturn model.chi2_ndf"


        #def do_min(a, b, c, d, e, f, g, h, i, j, k): #FIXME!!!
        #    model.startparams = (a, b, c, d, e, f, g, h, i, j, k)
        #    model.fit_to_data(charges, nbins, silent=True, **kwargs)
        #    return model.chi2_ndf
        exec(funcstring)
        bnd = kwargs["bounds"]
        if "bounds" in kwargs:
            min_kwargs = dict()
            for i,__ in enumerate(startparams):
                min_kwargs["limit_" + names[i]] =(bnd[0][i],bnd[1][i])
            m = Minuit(do_min, **min_kwargs)
            #m = Minuit(do_min, limit_a=(bnd[0][0],bnd[1][0]),
            #                   limit_b=(bnd[0][1],bnd[1][1]),
            #                   limit_c=(bnd[0][2],bnd[1][2]),
            #                   limit_d=(bnd[0][3],bnd[1][3]),
            #                   limit_e=(bnd[0][4],bnd[1][4]),
            #                   limit_f=(bnd[0][5],bnd[1][5]),
            #                   limit_g=(bnd[0][6],bnd[1][6]),
            #                   limit_h=(bnd[0][7],bnd[1][7]),
            #                   limit_i=(bnd[0][8],bnd[1][8]),
            #                   limit_j=(bnd[0][9],bnd[1][9]),
            #                   limit_k=(bnd[0][10],bnd[1][10]))
        else:



            m = Minuit(do_min)
        # hand over the startparams
        for key, value in zip(["a","b","c","d","e","f","g","h","i","j"], startparams):
            m.values[key] = value
        m.migrad()
    else:
        model.startparams = startparams
        model.fit_to_data(charges, nbins,normalize=normalize, silent=silent, **kwargs)

    # check for named tuple
    if hasattr(startparams, "_make"): # duck typing
        best_fit_params = startparams._make(model.best_fit_params)
    else:
        best_fit_params = model.best_fit_params
    print("Best fit parameters {}".format(best_fit_params))

    return model

############################################



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
