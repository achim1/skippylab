"""
Communicate with oscilloscope via vxi11 protocol over LAN network
"""
import abc
import time
import numpy as np
import pylab as p
import vxi11
from six import with_metaclass

from . import commands as cmd
from . import logging
from . import plotting
from . import tools

from copy import copy

bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    pass
    #logger.warning("No pyprind available")

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

from functools import reduce

# abbreviations
aarg = cmd.add_arg
q = cmd.query

TCmd = cmd.TektronixDPO4104BCommands
RSCmd = cmd.RhodeSchwarzRTO1044Commands

BIG_NUMBER = 1e25 # A number larger than the amount of captured samples

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


class AbstractBaseOscilloscope(with_metaclass(abc.ABCMeta, object)):
    """
    A oscilloscope with a high sampling rate in the order of several
    gigasamples. Defines the scope API the DAQ reiles on
    """

    def __init__(self, ip="169.254.68.19", loglevel=20):
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
        self.logger = logging.get_logger(loglevel)

    def reopen_socket(self):
        """
        Close and reopen the socket after a timeout

        Returns:
            None
        """
        self.logger.debug("Reopening socket on ip {}".format(self.ip))
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

        self.logger.debug("Sending {}".format(command))
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
        self.logger.debug("Sending {}".format(command))
        self.instrument.write(command)

    def ping(self):
        """
        Check if oscilloscope is connected
        """

        ping = self._send(cmd.WHOAMI)
        self.logger.info("Scope responds to {} with {}".format(cmd.WHOAMI, ping))
        return True if ping else False

    def __repr__(self):
        """
        String representation of the scope
        """
        ping = self._send(cmd.WHOAMI)
        return "<" + ping + ">"

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

    @abc.abstractproperty
    def samplingrate(self):
        """
        Get the current sampling rate

        Returns:
            float (GSamples/sec)
        """
        return

    def __del__(self):
        """
        Destructor, close connection explicitely.

        Returns:
            None
        """
        self.instrument.close()


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
    #fast_acquisition = setget(cmd.ACQUIRE_FAST_STATE)
    acquire = setget(cmd.RUN)
    acquire_mode = setget(TCmd.ACQUISITON_MODE)
    data = setget(cmd.DATA)
    trigger_frequency_enabled = setget(TCmd.TRIGGER_FREQUENCY_ENABLED)
    histbox = setget(cmd.HISTBOX)
    histstart = setget(cmd.HISTSTART)
    histend = setget(cmd.HISTEND)

    def __init__(self, ip, loglevel=20):
        AbstractBaseOscilloscope.__init__(self, ip, loglevel=loglevel)
        self.active_channel = TCmd.CH1
        self._header_buff = False
        self._wf_buff = np.zeros(len(self.waveform_bins))
        self._data_start_stop_buffer = (None, None)

        # FIXME: future extension
        self._is_running = False
        self._acquisition_single = False

        # fill the buffer
        self.fill_header_buffer()
        self.fill_buffer()

    def fill_header_buffer(self):
        self._header_buff = self.wf_header()

    def trigger_single(self):
        self.acquire_mode = cmd.RUN_SINGLE

    def trigger_continuous(self):
        self.acquire_mode = cmd.RUN_CONTINOUS
        self.acquire = cmd.START_ACQUISITIONS

    def _trigger_acquire(self):
        """
        Acquire one single waveform

        Returns:
            None
        """
        self.acquire = "ON"

    @staticmethod
    def _parse_wf_header(header):
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
        self.logger.info("Selecting channel {}".format(self.source))
        return None

    def _select_active_channel(self):
        """
        Pick the channel which is intended to be used

        Returns:

        """
        self.source = self.active_channel

    @property
    def samplingrate(self):
        """
        The samplingrate in GSamples/S

        Returns:
            float
        """
        head = self.wf_header()
        self.logger.debug("Got samplingrate of {}".format(1./head["xincr"]))
        return 1./head["xincr"]

    @property
    def trigggerrate(self):
        """
        The rate the scope is triggering. The scope in principle provides this number,
        however we have to work around it as it does not work reliably

        Keyword Arguments:
            interval (int): time interval to integrate measurement over in seconds

        Returns:
            float
        """
        self.logger.debug("The returned value is instantanious!\n "
                            "For serious measurements, gather some statistics!")
        self.trigger_frequency_enabled = TCmd.ON
        freq = float(self._send(TCmd.TRIGGER_FREQUENCYQ))
        return freq

    def reset_acquisition_window(self):
        """
        Reset the acquisition window to the maximum possible acquisition window
        Returns:In
            None
        """
        self.data_start = 0
        self.data_stop = BIG_NUMBER # temporarily set this to a big bogus number
                                    # this will result in the correct value for
                                    # "npoints" later
        head = self.wf_header()
        self.set_acquisition_window(0, head["npoints"])

    @property
    def time_binwidth(self):
        """
        Get the binwidth of the time - that is sampling rate

        Returns:
            float
        """
        head = self.wf_header()
        return float(head["xincr"])

    @property
    def waveform_bins(self):
        """
        Get the time bin numbers for the waveform voltage data

        Returns:
            np.ndarray
        """
        head = self.wf_header()
        bin_zero = self.data_start
        bins = np.linspace(int(self.data_start), int(self.data_stop), int(head["npoints"]))
        return bins

    @property
    def waveform_times(self):
        """
        Get the time for the waveform bins

        Returns:
            np.ndarray
        """
        head = self.wf_header()
        return head["xs"]

    @property
    def histogram(self):
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

    def wf_header(self, absolute_timing=False):
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

    def acquire_waveform(self, buff_header=False):
        """
        Get the waveform data

        Keyword Args:
            buff_header: buffer the header (do NOT query the header every time)
                         WARNING: This is only correct if there are no changes
                                  to the way the acquisition is made

        Returns:
            np.ndarray
        """

        waveform = self._send(cmd.CURVE)
        waveform = np.array([float(k) for k in waveform.split(",")])
        if buff_header:
            header = self._header_buff
            assert header is not None, "Nothing in header buffer, call fill_header_buffer"
        else:
            header = self.wf_header()

        # from the docs
        # Value in YUNit units = ((curve_in_dl - YOFf) * YMUlt) + YZEro
        waveform = (waveform - (np.ones(len(waveform))*header["yoff"]))\
                   *(np.ones(len(waveform))*header["ymult"])\
                   +(np.ones(len(waveform))*header["yzero"])

        return waveform

    def set_acquisition_window(self, start, stop):
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
        self._wf_buff = np.zeros(len(self.waveform_bins))
        self._data_start_stop_buffer = (start, stop)
        self.logger.info("Set acquisition window to {} - {}".format(self.data_start, self.data_stop))

    def set_acquisition_window_from_internal_buffer(self):
        """
        Use the internal buffer to set the data acquisition window. Might be necessary
        if the channel was switched in the meantime

        Returns:
            None
        """
        self.data_start, self.data_stop = self._data_start_stop_buffer
        #self.fill_header_buffer()
        #self.fill_bu

    def set_feature_acquisition_window(self, leading, trailing, n_waveforms=20):
        """
        Set the acquisition window around the most prominent feature in the waveform

        Args:
            leading (float): leading ns before the most prominent feature
            trailing (float): trailing ns after the most prominent feature

        Keyword Args
            n_waveforms (int): average over n_waveforms to identify the most prominent feature

        Returns:
            None
        """
        self.reset_acquisition_window()
        xs, avg = self.average_waveform(n=n_waveforms)
        wf_bins = self.waveform_bins

        abs_avg = abs(avg)
        feature_y = max(abs_avg)
        #feature_x = xs[abs_avg == feature_y]
        feature_x_bin = wf_bins[abs_avg == feature_y]
        bin_width = self.time_binwidth
        leading_bins = 1e-9*float(leading)/bin_width
        trailing_bins = 1e-9*float(trailing)/bin_width
        data_start = int(feature_x_bin - leading_bins)
        data_stop = int(feature_x_bin + trailing_bins)
        self.set_acquisition_window(data_start, data_stop)
        return None

    def make_n_acquisitions(self, n,\
                            trials=20, return_only_charge=False,\
                            single_acquisition=True):
        """
        Acquire n waveforms

        Args:
            n (int): Number of waveforms to acquire

        Keyword Args:
            trials (int): Set breaking condition when to abort acquisition
            return_only_charge (bool): don't get the wf, but only integrated charge instead
            single_acquisition (bool): use the scopes single acquisition mode

        Returns:
            list: [wf_1,wf_2,...]

        """
        wforms = list()
        acquired = 0
        trial = 0
        if bar_available:
            bar = pyprind.ProgBar(n, track_time=True, title='Acquiring waveforms...')
        if single_acquisition:
            self.trigger_single()
        else:
            self.trigger_continuous()
        wf_buff = 0
        while acquired < n:
            try:
                if single_acquisition:
                    self.acquire = "ON"
                wf = self.acquire_waveform(buff_header=True)
                if (wf[0]*np.ones(len(wf)) - wf).sum() == 0:
                    continue # flatline test
                if (wf - wf_buff).sum() == 0:
                    continue # test if scope just returned the
                             # same waveform again
                if return_only_charge:
                     wf = tools.integrate_wf(header, wf)
                wf_buff = wf
                wforms.append(wf)
                acquired += 1
                if bar_available:
                    bar.update()

            except Exception as e:
                self.logger.critical("Can not acquire wf..{}".format(e))
                trial += 1
            if trial == trials:
                break
        if bar_available:
            print(bar)
        return wforms

    def average_waveform(self,n=10):
        """
        Acquire some waveforms and take the average

        Keyword Args.
            n (int): number of waveforms to average over

        Returns:
            tuple(np.array). xs, ys

        """

        wf = self.make_n_acquisitions(n, single_acquisition=False)
        xs = self.waveform_times
        len_wf = [len(w) for w in wf]
        wf = [w for w in wf if len(w) == min(len_wf)]
        avg = reduce(lambda x, y: x+y, wf)/n
        return xs, avg

    def show_waveforms(self,n=5):
        """
        Demonstration function: Will use pylab show to
        plot some acquired waveforms

        Keyword Args:
            n (int): number of waveforms to show

        Returns:
            None
        """

        wf = self.make_n_acquisitions(n, single_acquisition=False)
        head = self.wf_header()
        for i in range(len(wf)):
            try:
                plotting.plot_waveform(head, wf[i])
            except Exception as e:
                print(e)

        p.show()

    def fill_buffer(self):
        """


        Returns:

        """
        self._wf_buff = self.acquire_waveform()


    def pull(self, buff_header=True, use_buffered_acq_window=True,
             use_channel_info=True):
        """
        Fit in the API for the DAQ. Returns waveform data

        Keyword Args:
            buff_header (bool): buffer the header for subsequent acquisition without
                                changing the parameters of the acquistion (much faster)
            FIXME! Default value of this should be False, however requires DAQ API change
            use_buffered_acq_window (bool): set this flag to cache the length of the acquisition window
                                            internally so that it does not get resetted when switching channels
            use_channel_info (bool): select the channel on each submit
        Returns:
            dict
        """
        user_error_msg = """ This pull method is designed to be used in single acquisition mode."""

        assert self.acquire_mode == cmd.RUN_SINGLE, user_error_msg

        # FIXME: The buffer mechanism fails if this is the first
        # waveform at all.
        data = dict()
        if use_channel_info:
            self._select_active_channel()
        if use_buffered_acq_window:
            self.set_acquisition_window_from_internal_buffer()
        #while True:
        wf = self.acquire_waveform(buff_header=buff_header)

            #if (wf[0]*np.ones(len(wf)) - wf).sum() == 0:
            #    continue # flatline test
            #elif (wf - self._wf_buff).sum() == 0:
            #    self._wf_buff = wf
            #    continue # test if scope just returned the
            #             # same waveform again
            #else:
            #    self._wf_buff = wf
            #    break
        if buff_header:
            data.update(self._header_buff)
        else:
            data.update(self.wf_header())
        data["waveform"] = wf
        return data


class RhodeSchwarzRTO1044(AbstractBaseOscilloscope):
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
