"""
Communicate with oscilloscope via vxi11 protocoll over LAN network
"""


import abc
import time
import numpy as np
import vxi11
from six import with_metaclass

from . import commands as cmd
from . import logging

from copy import copy

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

# abbreviations
#dec = cmd.decode
#enc = cmd.encode
aarg = cmd.add_arg
q = cmd.query

TCmd = cmd.TektronixDPO4104BCommands
RSCmd = cmd.RhodeSchwarzCommands

def setget(command):
    """
    Shortcut to construct property object to wrap getters and setters
    for a number of settings

    Args:
        command (str): The command being used to get/set. Get will be a query
        value (str): The value to set

    Returns:
        property object
    """
    return property(lambda self: self._send(q(command)),\
                    lambda self, value: self._set(aarg(command, value)))


class AbstractBaseOscilloscope(with_metaclass(abc.ABCMeta,object)):
    """
    A oscilloscope with a high sampling rate in the order of several
    gigasamples. Defines the scope API the DAQ reiles on
    """

    def __init__(self, ip="169.254.68.19"):
        """
        Connect to the scope via its socket server

        Args:
            ip (str): ip of the scope
        """
        self.ip = ip
        self.connect_trials = 0
        self.wf_buff_header = None # store a waveform header in case they are all the same
        self.instrument = vxi11.Instrument(ip)
        self.active_channel = None

    def reopen_socket(self):
        """
        Close and reopen the socket after a timeout

        Returns:
            None
        """
        self.instrument = vxi11.Instrument(self.ip)

    def _send(self,command):
        """
        Send command to the scope. Raises socket.timeout error if
        it had failed too often

        Args:
            command (str): command to be sent to the
                           scope

        """
        if self.connect_trials == self.MAXTRIALS:
            self.connect_trials = 0
            raise vxi11.vxi11.Vxi11Exception("TimeOut")

        if self.verbose: print ("Sending {}".format(command))
        try:
            response = self.instrument.ask(command)
        except Exception as e:
            self.reopen_socket()
            response = self.instrument.ask(command)
            self.connect_trials += 1

        return response

    def _set(self, command):
        """
        Send a command bur return no response

        Args:
            command (str): command to be send to the scope

        Returns:
            None
        """
        if self.verbose: print("Sending {}".format(command))
        self.instrument.write(command)

    def ping(self):
        """
        Check if oscilloscope is connected
        """

        ping = self._send(cmd.WHOAMI)
        print (ping)
        return True if ping else False

    def __repr__(self):
        """
        String representation of the scope
        """
        return "<" + self._send(cmd.WHOAMI) + ">"

    @abc.abstractmethod
    def select_channel(self, channel):
        """
        Select a channel for the data acquisition

        Args:
            channel (int): Channel number

        Returns:
            None
        """
        return


class Waveform(object):
    """
    A non-oscilloscope dependent representation of a measured
    waveform
    """
    header = dict()

    def __init__(self, header, curvedata):
        """
        Args:
            header (dict): Metadata, like xs, units, etc.
            curvedata (np.ndarray): The voltage data
        """

        self.header = header
        self.data = curvedata

    @property
    def ns(self):
        """
        Return the signal recording time with ns precision

        Returns:
            np.ndarray
        """
        return

    @property
    def mV(self):
        """
        Return the recorded voltages per bin

        Returns:
            np.ndarray
        """
        return

    @property
    def bins(self):
        """
        Return digitizer time bin numbers (arbitrary scale)

        Returns:
            np.ndarray
        """
        return


class TektronixDPO4104B(AbstractBaseOscilloscope):
    """
    Oscilloscope of type DPO4104B manufactured by Tektronix
    """

    # constants used by the socket connection
    MAXTRIALS = 5

    # setget properties
    source = setget(cmd.SOURCE)
    data_start = setget(cmd.DATA_START)
    data_stop = setget(cmd.DATA_STOP)
    waveform_enc = setget(cmd.WF_ENC)
    fast_acquisition = setget(cmd.ACQUIRE_FAST_STATE)
    acquire = setget(cmd.RUN)
    acquire_mode = setget(cmd.ACQUIRE_STOP)
    data = setget(cmd.DATA)
    histbox = setget(cmd.HISTBOX)
    histstart = setget(cmd.HISTSTART)
    histend = setget(cmd.HISTEND)
    verbose = False

    def __init__(self, ip):
        AbstractBaseOscilloscope.__init__(self, ip)
        self.active_channel = TCmd.CH1

    def _parse_wf_header(self,header):
        """
        Parse a waveform header send by our custom WF_HEADER command
        The reason why we are not using WFM:Outpre is that the documentation
        was not so sure about how its response might look

        Args:
            head (str): the result of a WF_HEADER command

        Returns:
            dict
        """
        head = header.split(";")
        keys = ["bytno", "enc", "npoints", "xzero", "xincr", "yzero", "yoff", \
                "ymult", "xunit", "yunit"]

        assert len(head) == len(keys), "Cannot read out all the header info I want!"

        parsed = dict(zip(keys, head))
        for k in parsed:
            try:
                f = float(parsed[k])
                parsed[k] = f
            except ValueError:
                continue

            # get rid of extra " in units
            parsed["xunit"] = parsed["xunit"].replace('"', '')
            parsed["yunit"] = parsed["yunit"].replace('"', '')

            # also some are ints
            parsed["npoints"] = int(parsed["npoints"])
        return parsed

    def select_channel(self, channel):
        """
        Select the channel for the readout

        Args:
            channel (int): Channel number (1-4)

        Returns:
            None
        """

        assert 0 < channel < 5, "Channel value has to be 1-4"

        channel_dict = {1 : TCmd.CH1, 2: TCmd.CH2, 3: TCmd.CH3, 4: TCmd.CH4}
        self.source = channel_dict[channel]
        self.active_channel = channel_dict[channel]
        return

    def get_triggerrate(self):
        """
        The rate the scope is triggering. This number is provided
        by the scope. Most times it is nan though...

        Returns:
            float
        """
        trg_rate = self._send(cmd.TRG_RATEQ)
        trg_rate = float(trg_rate)
        # from the osci docs
        # the IEEE Not A Number (NaN = 99.10E+36)
        if trg_rate > 1e35:
            trg_rate = np.nan
        return trg_rate

    def reset_acquisition_window(self):
        """
        Reset the acquisition window to some factory default reasonables

        Returns:
            None
        """
        self.data = cmd.SNAP

    def get_time_binwidth(self):
        """
        Get the binwidth of the time - that is sampling rate

        Returns:
            float
        """
        head = self.get_wf_header()
        return float(head["xincr"])

    def get_waveform_bins(self):
        """
        Get the time bin numbers for the waveform voltage data

        Returns:
            np.ndarray
        """
        head = self.get_wf_header()
        bin_zero = self.data_start
        bins = np.linspace(int(self.data_start), int(self.data_stop), int(head["npoints"]))
        return bins

    def get_waveform_times(self):
        """
        Get the time for the waveform bins

        Returns:
            np.ndarray
        """
        head = self.get_wf_header()
        return head["xs"]

    def get_histogram(self):
        """
        Return a histogram which might be recorded
        by the scope
        """
        start = self.histstart
        end = self.histend
        bincontent = self._send(cmd.HISTDATA)
        assert None not in [start,end,bincontent],\
                   "Try again! might just be a hickup {} {} {}".format(start,end,bincontent)

        bincontent = np.array([int(b) for b in bincontent.split(",")])#
        start = float(start)
        end = float(end)    
        nbins = len(bincontent)
        if start > end:
            print ("Swapping start and end...")
            tmpstart = copy(start)
            start = end
            end = tmpstart
            del tmpstart

        print("Found histogram with {} entries from {:4.2e} to {:4.2e}".format(nbins,start, end))

        l_binedges = np.linspace(start,end,nbins + 1)[:-1]
        r_binedges = np.linspace(start,end,nbins + 1)[1:]
        bincenters = r_binedges + (r_binedges - l_binedges)/2.
        return bincenters, bincontent 

    def get_wf_header(self, absolute_timing=False):
        """
        Get some meta information about the *next incoming wavefrm*

        Keyword Args:
            absolute_timing (bool): header["xs"] starts with header["xzero"], 0 otherwiese

        Returns:
            dict
        """
        header = self._send(TCmd.WF_HEADER)
        header = self._parse_wf_header(header)
        self.wf_buff_header = header
        #header["xs"] = np.ones(len(header["npoints"]))*header["xzero"]
        if absolute_timing:
            xs = np.ones(header["npoints"])*header["xzero"]
        else:
            # relative timing?
            xs = np.zeros(int(header["npoints"]))
        # FIXME: There must be a better way
        for i in range(int(header["npoints"])):
            xs[i] += i*header["xincr"]

        header["xs"] = xs
        return header

    def get_waveform(self, single_acquisition=False):
        """
        Get the waveform data


        Args:
            single_acquire: use single acquition mode

        Returns:

        """
        #self.acquire_mode = cmd.SINGLE_ACQUIRE
        if single_acquisition: self.acquire = cmd.ON
        waveform = self._send(cmd.CURVE)
        if single_acquisition: self.acquire = cmd.OFF
        waveform = np.array([float(k) for k in waveform.split(",")])
        header = self.get_wf_header()

        # from the docs
        # Value in YUNit units = ((curve_in_dl - YOFf) * YMUlt) + YZEro
        waveform = (waveform - (np.ones(len(waveform))*header["yoff"]))\
                   * (np.ones(len(waveform))*header["ymult"])\
                   + (np.ones(len(waveform))*header["yzero"])

        return waveform

    def set_single_acquistion(self):
        """
        Set the scope in single acquisition mode

        Returns:
            None
        """
        self.acquire = cmd.SINGLE_ACQUIRE

    def set_acquisition_window(self,start, stop):
        """
        Set the acquisition window in bin number

        Args:
            start (int): start bin
            stop (int): stop bin

        Returns:
            None
        """
        self.data_start = start
        self.data_stop = stop

    def get_acquisition_window_start(self):
        """
        Return the values of the acquisition window in bins

        Returns:
            tuple (int, int)
        """

        return int(self.data_start), int(self.data_stop)


class RhodeSchwarzRTO(AbstractBaseOscilloscope):
    """
    Made by Rhode&Schwarz, scope with sampling rate up to 20GSamples/s
    """

    def __init__(self, ip):
        AbstractBaseOscilloscope.__init__(self,ip)
        self.active_channel = RSCmd.CH1

    def select_channel(self, channel):
        """
        Select the channel for the readout

        Args:
            channel (int): Channel number (1-4)

        Returns:
            None
        """
        channel_dict = {1: RSCmd.CH1, 2: RSCmd.CH2, 3: RSCmd.CH3, 4: RSCmd.CH4}
        self.active_channel = channel_dict[channel]

    def get_waveform(self):
        """
        Get the voltage values for a single waveform

        Returns:
            np.ndarray
        """
        wf_commnad = aarg(self.active_channel,RSCmd.CURVE)
        raw_wf = self._send(wf_command)
        pairs = izip(*[iter(raw_wf.split(","))]*2)
        times, volts = [],[]
        for val in pairs:
            volts.append(float(val[1]))
        return np.array(volts)