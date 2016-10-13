"""
Convenient plot functions

"""

import pylab as p
import seaborn.apionly as sb

from copy import copy

p.style.use("pyoscidefault")


def plot_waveform(wf_header, wf_data,\
                  fig=None,savename=None,\
                  use_mv_and_ns=True,color=sb.color_palette("dark")[0]):
    """
    Make a plot of a single acquisition

    Args:
        wf_header (dict): custom waveform header
        wf_data (np.ndarray): waveform data

    Keyword Args:
        fig (pylab.figure): A figure instance
        savename (str): where to save the figure (full path)
        use_mv_and_ns (bool): use mV and ns instead of V and s
    Returns:
        pylab.fig
    """

    if fig is None:
        fig = p.figure()
    ax = fig.gca()

    # if remove_empty_bins:
    #    bmin = min(bincenters[bincontent > 0])
    #    bmax = max(bincenters[bincontent > 0])
    #    bincenters = bincenters[np.logical_and(bincenters >= bmin, bincenters <= bmax)]
    #    bincontent = bincontent[np.logical_and(bincenters >= bmin, bincenters <= bmax)]

    xlabel = wf_header["xunit"]
    ylabel = wf_header["yunit"]
    xs = copy(wf_header["xs"])
    ys = copy(wf_data)

    if xlabel == "s" and ylabel == "V" and use_mv_and_ns:
        xs *= 1e9
        ys *= 1e3
        xlabel = "ns"
        ylabel = "mV"
    ax.plot(xs, ys, color=color)
    ax.grid()
    sb.despine(fig)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    p.tight_layout()
    if savename is not None:
        fig.savefig(savename)
    return fig

#####################################################

def plot_histogram(bincenters,bincontent,\
                   fig=None,savename="test.png",\
                   remove_empty_bins=True):
    """
    Plot a histogram returned by TektronixDPO4104B.get_histogram
    Use pylab.plot

    Args:
        bincenters (np.ndarray); bincenters (x)
        bincontent (np.ndarray): bincontent (y)

    Keyword Args:
        fig (pylab.figure): A figure instance
        savename (str): where to save the figure (full path)
        remove_empty_bins (bool): Cut away preceeding and trailing zero bins


    """


    if fig is None:
        fig = p.figure()
    ax = fig.gca()
    if remove_empty_bins:
        bmin = min(bincenters[bincontent > 0])
        bmax = max(bincenters[bincontent > 0])
        bincenters = bincenters[np.logical_and(bincenters >= bmin, bincenters <= bmax)]
        bincontent = bincontent[np.logical_and(bincenters >= bmin, bincenters <= bmax)]

    ax.plot(bincenters,bincontent,color=sb.color_palette("dark")[0])
    ax.grid()
    sb.despine(fig)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("log nevents ")
    p.tight_layout()
    fig.savefig(savename)
    return fig


