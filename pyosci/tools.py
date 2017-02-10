"""
Convenient operations

"""

import numpy as np
import scipy.integrate as integrate
from scipy.constants import elementary_charge as ECHARGE
from copy import deepcopy as copy

IMPEDANCE = 50


def average_wf(waveforms):
    """
    Get the average waveform

    Args:
        waveforms (list):

    Returns:
        np.ndarray
    """
    wf0 = copy(waveforms[0])
    for wf in waveforms[1:]:
        wf0 += wf

    return wf0 / float(len(waveforms))


def integrate_wf(header, waveform, method=integrate.simps, impedance = IMPEDANCE):
    """
    Integrate a waveform to get the total charge

    Args:
        header (dict):
        waveform (np.ndarray):

    Returns:
        float
    """
    integral = method(waveform, header["xs"], header["xincr"])
    return integral/impedance


def save_waveform(header, waveform, filename):
    """
    save a waveform together with its header

    Args:
        header (dict):
        waveform (np.ndarray):
        filename (str):
    Returns:
        None
    """
    np.save(filename, (header, waveform))
    return None


def load_waveform(filename):
    """
    load a waveform from a file

    Args:
        filenaame (str): 

    Returns:
        dict
    """
    head, wf = np.load(filename)
    return head, wf
