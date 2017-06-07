"""
Communicate with oscilloscope via vxi11 protocol over LAN network

"""
import abc
import time
import numpy as np
import pylab as p
import vxi11
import re
import struct
from six import with_metaclass

from .. scpi import commands as cmd
from .. import loggers
from .. import plotting
from .. import tools

from copy import copy

bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    pass

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
    # constants used by the socket connection
    MAXTRIALS = 5
    CONTINOUS_RUN = cmd.RUN_CONTINOUS
    ACQUIRE_ONE = cmd.RUN_SINGLE

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
        self.logger = loggers.get_logger(loglevel)

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

    # def run(self):
    #     """
    #     Start data acquisition
    #
    #     Returns:
    #         None
    #     """
    #     self.logger.debug("Starting data acquisition")
    #     assert self.acquire_mode == cmd.RUN_CONTINOUS,\
    #                                 "Run is ment to use with continous acquisition!Set self.acquire_mode accordingly"
    #     self.acquire = cmd.START_ACQUISITIONS
    #
    # def stop(self):
    #     """
    #     Stop any ongoing data acquisition
    #
    #     Returns:
    #         None
    #     """
    #     self.acquire = cmd.STOP_ACQUISITIONS
    #
    # def do_single_acquisition(self):
    #     """
    #     Acquire a single event
    #
    #     Returns:
    #         None
    #     """
    #     assert self.acquire == cmd.RUN_SINGLE, "Set scope to single acquistion mode first!"
    #     return self.acquire_waveform()
    #
    # def do_single_acquisition_fast(self):
    #     """
    #     Acquire a single event. FAST mode (no check)
    #
    #     Returns:
    #
    #     """
    #     return self.acquire_waveform()

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

    @abc.abstractmethod
    def acquire_waveform(self):
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

    @property
    def header(self):
        pass


class UnknownOscilloscope(AbstractBaseOscilloscope):
    """
    Use for testing and debugging

    """
    def select_channel(self, channel):
        raise NotImplementedError("Not implemented!")

    def samplingrate(self):
        raise NotImplementedError("Not Implemented!")

    def acquire_waveform(self):
        raise NotImplementedError("Not Implemented!")

# getters/setters for the header
def get_header(self):
    if not self._header:
        response = self.send(TCmd.WF)
        self._header = self.decode_header(response)

    return self._header


def set_header(self, header):
    self._header = header


class TektronixDPO4104B(AbstractBaseOscilloscope):
    """
    Oscilloscope of type DPO4104B manufactured by Tektronix
    """

    # setget properties
    source = setget(cmd.SOURCE)
    data_start = setget(cmd.DATA_START)
    data_stop = setget(cmd.DATA_STOP)
    waveform_enc = setget(cmd.WF_ENC)
    acquire = setget(cmd.RUN)
    acquire_mode = setget(TCmd.ACQUISITON_MODE)

    data = setget(cmd.DATA)
    trigger_frequency_enabled = setget(TCmd.TRIGGER_FREQUENCY_ENABLED)
    histbox = setget(cmd.HISTBOX)
    histstart = setget(cmd.HISTSTART)
    histend = setget(cmd.HISTEND)

    # FIXME make it a property
    binary_formats = {"RI": "!b"} # transform the binary format to something
                                # which is understandable by the struct
                                # module
    binary_header_pattern = re.compile("#(?P<bin_head>[0-9]*)")

    def __init__(self, ip, loglevel=20):
        AbstractBaseOscilloscope.__init__(self, ip, loglevel=loglevel)
        self.active_channel = TCmd.CH1
        self._data_start_stop_buffer = (None, None)
        self.header = property(get_header, set_header)

        # FIXME: future extension
        self._is_running = False
        self._acquisition_single = False

        # prepare the scope - use binary encoding by default
        #self.waveform_enc = cmd.ASCII
        self._set("DATa:ENCdg FAST")

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
    def decode_header(response, return_last_index=False, absolute_timing=False):
        """
        Parse a response searching for waveform header data

        Args:
            head (str): hopefully the result of some WAVFrm or similar command

        Keyword Args:
            return_last_index (bool): if True, also the last index of the header in
                                      the string will be returned
            absolute_timing (bool) : try to infer the absolute timeing (whatever that means)
                                     # FIXME!
        Returns:
            dict/tuple
        """
        # create patterns
        pattern = '(?P<byteno>\d*);(?P<bitno>\d*);(?P<enc>\D*);(?P<bnform>\D*);(?P<bitor>\D*);'
        pattern += '(?P<wfid>[a-zA-Z0-9,./"\s]*);(?P<npoints>\d*);(?P<pointfmt>\D*);'
        pattern += '(?P<xunit>[A-Za-z0-9\"]*);(?P<xincr>[0-9\.E\-\+]*);(?P<xzero>[0-9\.E\-\+]*);(?P<ptoff>\d*);'
        pattern += '(?P<yunit>[A-Za-z0-9\"]*);(?P<ymult>[0-9\.E\-\+]*);(?P<yoff>[0-9\.E\-\+]*);'
        pattern += '(?P<yzero>[0-9\.E\-\+]*);'

        wfid_subpat = '"(?P<channel>[A-Za-z0-9]*),\s*(?P<cpling>[A-Za-z0-9\s]*),\s*'
        wfid_subpat += '(?P<vdiv>[A-Za-z0-9./]*),\s*(?P<hdiv>[A-Za-z0-9./]*),\s*'
        wfid_subpat += '(?P<points>[A-Za-z0-9./\s]*),\s*(?P<sampmode>[A-Za-z0-9./\s]*);?"'

        subregex = re.compile(wfid_subpat)
        regex = re.compile(pattern)

        parsed = regex.search(response)
        header = {}
        if parsed is None:
            return header

        header.update(parsed.groupdict())
        wfid_parsed = subregex.search(header["wfid"])
        if wfid_parsed is not None:
            header.pop("wfid")
            header.update(wfid_parsed.groupdict())

        # now convert the fields
        for k in header:
            try:
                header[k] = float(header[k])
            except ValueError:
                #shouganei ne
                pass

        # get rid of extra " in units
        header["xunit"] = header["xunit"].replace('"', '')
        header["yunit"] = header["yunit"].replace('"', '')

        # also some are ints
        header["npoints"] = int(header["npoints"])
        header["byteno"] = int(header["byteno"])

        if absolute_timing:
            xs = np.ones(header["npoints"])*header["xzero"]
        else:
            # relative timing?
            xs = np.zeros(int(header["npoints"]))

        # FIXME: There must be a better way
        for i in range(int(header["npoints"])):
            xs[i] += i*header["xincr"]

        header["xs"] = xs

        last_index = 0
        if return_last_index:
            __, last_index = parsed.span(parsed.lastgroup)
            return header, last_index
        return header

    def set_waveform_encoding(self, enc):
        """
        Define the waveform encoding

        Args:
            enc:

        Returns:

        """
        raise NotImplementedError("Not implemented!")

    @staticmethod
    def decode_ascii_waveform(response):
        """
        Search the response for waveform data when in ascii format

        Args:
            response (str): Hopefully the result of a CURVE command or similar

        Returns:
            np.ndarray
        """
        response = response.replace(";", "") # clean trailing ;
        data = np.fromstring(response, sep=",")
        return data

    @staticmethod
    def decode_binary_waveform(response, header):
        """
        Decaode a waveform in binary format. To do so, the header is
        required to know about the exact format.

        Args:
            response (str): Hopefully the response to some CURVE command or similar
            header (dict): A parsed waveform header

        Returns:
            np.ndarray
        """
        bin_format = TektronixDPO4104B.binary_formats[header["bnform"]]
        # FIXME: should not be necessary
        if response.startswith(";"):
            response = response[1:]

        #split of header first
        bin_header = TektronixDPO4104B.binary_header_pattern.search(response)
        last_index = bin_header.span(bin_header.lastgroup)[1]

        endianess = "!" # msd first
        if header["bnform"].startswith("S"):
            endianess = "<" # little endian

        if header["byteno"] == 1:
            bin_format = endianess + "b"
        elif header["byteno"] == 2:
            bin_format = endianess + "h"
        else:
            raise ValueError("Unsupported binary format! Please check header!")

        buffer = struct.iter_unpack(bin_format, bytes(response[last_index:].encode("utf-8")))
        data = np.array([i[0] for i in buffer])
        return data

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
        self.logger.debug("Got samplingrate of {}".format(1./self.header["xincr"]))
        return 1./self.header["xincr"]

    @property
    def triggerrate(self):
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
        self.set_acquisition_window(0, self.header["npoints"])

    @property
    def time_binwidth(self):
        """
        Get the binwidth of the time - that is sampling rate

        Returns:
            float
        """
        head = self.header
        return float(self.header["xincr"])

    @property
    def waveform_bins(self):
        """
        Get the time bin numbers for the waveform voltage data

        Returns:
            np.ndarray
        """
        bins = np.linspace(int(self.data_start), int(self.data_stop),\
                           int(self.header["npoints"]))
        return bins

    @property
    def waveform_times(self):
        """
        Get the time for the waveform bins

        Returns:
            np.ndarray
        """
        return self.header["xs"]

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

    def acquire_waveform(self, header=None):
        """
        Get the waveform data

        Keyword Args:
            header (dict): if header is None, a new header will be acquired

        Returns:
            np.ndarray
        """
        if header is None:
            wf_response = self._send(TCmd.WF)
            header, last_index = self.decode_header(wf_response, return_last_index=True)
            wf_data = wf_response[last_index:]
        else:
            wf_data = self._send(TCmd.WF_NOHEAD)

        # explicitely save the last header
        self.header = header
        if self.waveform_enc == cmd.ASCII or self.waveform_enc == "ASC":
            waveform = self.decode_ascii_waveform(wf_data)
        else:
            waveform = self.decode_binary_waveform(wf_data, header)
        # from the docs
        # Value in YUNit units = ((curve_in_dl - YOFf) * YMUlt) + YZEro
        waveform = ((waveform - (np.ones(len(waveform))*header["yoff"]))\
                    *(np.ones(len(waveform))*header["ymult"]))\
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
        # get the first waveform
        wf_buff = self.acquire_waveform()
        n -= 1
        while acquired < n:
            try:
                if single_acquisition:
                    self.acquire = "ON"
                wf = self.acquire_waveform(header=self.header)

                # flatline test
                if (wf[0]*np.ones(len(wf)) - wf).sum() == 0:
                    continue
                if (wf - wf_buff).sum() == 0:
                    continue # test if scope just returned the
                             # same waveform again
                if return_only_charge:
                     wf = tools.integrate_wf(self.header, wf)
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

        wf = self.make_n_acquisitions(n, single_acquisition=True)
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
        #head = self.wf_header()
        head = self.header
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
        #if buff_header:
        #    header = self.
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
        self.run_start_time = None

    def select_channel(self, channel):
        """
        Select the channel for the readout.

        Args:
            channel (int): Channel number (1-4)

        Returns:
            None
        """
        channel_dict = {1: RSCmd.CH1, 2: RSCmd.CH2, 3: RSCmd.CH3, 4: RSCmd.CH4}
        self.active_channel = channel_dict[channel]

    def acquire_waveform(self):
        """
        Get the voltage values for a single waveform

        Returns:
            np.ndarray
        """
        wf_command = aarg(self.active_channel,RSCmd.CURVE)
        raw_wf = self._send(wf_command)
        pairs = izip(*[iter(raw_wf.split(","))]*2)
        times, volts = [],[]
        for val in pairs:
            volts.append(float(val[1]))
        return np.array(volts)

    @property
    def samplingrate(self):
        raise NotImplementedError

    def run(self):
        """
        Start continuous acquisitions

        Returns:

        """
        self._set(RSCmd.RUN)
        self.run_start_time = time.monotonic()

    def stop(self):
        self._set(RSCmd.STOP)

    def do_single_acquisition(self):
        self._set(RSCmd.SINGLE)

    @property
    def triggerrate(self):
        """
        Get the triggerrate of the scope

        Args:
            interval (float): measurement time in seconds to

        Returns:
            float
        """

        nacq = self._send(RSCmd.N_ACQUISITONS)
        interval = time.monotonic() - self.run_start_time
        return float(nacq/interval)

