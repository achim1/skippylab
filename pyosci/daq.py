"""
Use the scope as a DAQ

"""
from . import osci
from . import tools
from . import commands as cmd
from . import plotting

import time
import numpy as np
import pylab as p

from socket import timeout as TimeoutError

bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    print ("No pyprind available")

try:
    from functools import reduce
except ImportError:
    print ("Can not import functools, this might be python 2.7?")




ACQTIME = .2

class DAQ(object):
    """
    A virtual DAQ using an Tektronix scope
    """

    def __init__(self,ip,port):
        """
        Initialize with a scope on ip listening at port

        Args:
            ip (str):
            port (int):
        """
        self.scope = osci.TektronixDPO4104B(ip,port)
        self._wf_header = None

        trials = 0
        ping = False
        while trials < 3 and not ping:
            ping = self.scope.ping()
            time.sleep(.1)
            trials += 1

        if not ping:
            raise TimeoutError

    @property
    def wf_header(self):
        """
        Get the current waveform header

        Returns:
            dict
        """
        #if self._wf_header is None:
        #    self._wf_header = self.scope.get_wf_header()
        return self.scope.get_wf_header()

    def set_acquisition_window_from_histbox(self):
        """
        Set the acquisition window for the waveforms
        following the histogram box on the screen

        Returns:
            None
        """
        self.scope.data = cmd.SNAP
        left, __, right, __ = self.scope.histbox.split(",")
        left = float(left)
        right = float(right)
        head = self.scope.get_wf_header()
        start, stop = 0, 0
        for k, val in enumerate(head["xs"]):
            if val >= left:
                start = k
                break

        for j, val in enumerate(head["xs"]):
            if val >= right:
                stop = j
                break
        print (start, stop)
        start += float(self.scope.data_start)
        stop += start
        print (start, stop)
        self.set_acq_window(start, stop)

    def setup(self):
        """
        Setup single acquisition and define the data window

        Returns:
            None
        """
        self.scope.data = cmd.SNAP
        self.scope.acquire = cmd.ACQUIRE_STOP
        self.scope.acquire_mode = cmd.RUN_SINGLE

    def set_acq_window(self, start, stop):
        """
        Setup the acquisition window in data points

        Args:
            start (int):
            stop (int):

        Returns:
            None
        """
        self.scope.data_start = str(start)
        self.scope.data_stop = str(stop)

    def get_acquisition_window(self):
        """
        Get the window the waveform is reconded in bins
        from the osci

        Returns:
            tuple
        """
        start = self.scope.data_start
        stop = self.scope.data_stop
        return (int(start), int(stop))

    def acquire_waveform(self):
        """
        Make an acquisition

        Returns:
            header, waveform
        """
        self.scope.acquire = cmd.SINGLE_ACQUIRE
        time.sleep(ACQTIME)
        wf = self.scope.get_waveform()
        return self.wf_header, wf

    def make_n_acquisitions(self, n, extra_timeout=.1,\
                            trials=20, return_only_charge=False,\
                            single_acquisition=True):
        """
        Acquire n waveforms

        Args:
            n (int): Number of waveforms to acquire

        Keyword Args:
            extra_timeout (float): Time between acquisitions
            trials (int): Set breaking condition when to abort acquisition
            return_only_charge (bool): don't get the wf, but only integrated charge instead
            single_acquisition (bool): use the scopes single acquisition mode

        Returns:
            list: [wf_1,wf_2,...]

        """
        wforms = list()
        acquired = 0
        trial = 0
        header = self.wf_header
        if bar_available:
            bar = pyprind.ProgBar(n, track_time=True, title='Acquiring waveforms...')
        if single_acquisition:
            self.scope.acquire_mode = cmd.RUN_SINGLE
        else:
            self.scope.acquire_mode = cmd.RUN_CONTINOUS
            self.scope.acquire = cmd.ON
        wf_buff = np.zeros(len(header["xs"]))
        while acquired < n:
            try:
                # self.scope.acquire = cmd.SINGLE_ACQUIRE
                # time.sleep(ACQTIME)
                wf = self.scope.get_waveform(single_acquisition=single_acquisition)
                # self.scope.acquire = cmd.SINGLE_ACQUIRE
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

                #time.sleep(ACQTIME)
            except Exception as e:
                print ("Can not acquire wf..")
                print (e.__repr__())
                trial += 1
                #time.sleep(ACQTIME)
            if trial == trials:
                break
        if bar_available:
            print (bar)
        return wforms

    def find_best_acquisition_window(self, leading=20,trailing=40,waveforms=20, offset=-1):
        """
        Takes an average waveform and identifies the peak location

        Keyword Args:
            leading (int): Append leading nanoseconds before leading edge
            trailing (int): Apppend trailing nanoseconds after falling edge
            waveforms (int): How many waveforms to average over
            offset (float): will be subtracted from baselinf for time over threshold

        Returns:
            tuple
        """
        self.scope.data = cmd.SNAP
        data_start = int(self.scope.data_start)
        data_stop = int(self.scope.data_stop)
        print ("Found acquisition window of {} {}".format(data_start, data_stop))
        xs, avg = self.get_average_waveform(n=waveforms)
        # find min
        wf_re = min(avg)
        wf_fe = -np.inf
        wf_re_y = 0
        wf_fe_y = 0
        new_data_start = data_start
        new_data_stop = data_stop
        min_found = False
        # buffer header
        head = self.scope.get_wf_header()
        wfmin = np.inf
        wfxmin = 0
        for xmin, val in enumerate(avg):
            if val < wfmin:
                wfmin = val
                wfxmin = xmin

        new_data_start = data_start + wfxmin - ((1e-9)*leading/head["xincr"])
        new_data_stop = data_start + wfxmin + ((1e-9)*trailing/head["xincr"])
        # find peak start-end with time over threshold
        #smoothed = np.gradient(avg)
        #threshold = np.gradient(smoothed) - np.ones(len(smoothed))*offset # .1 for smoothing
        #for i,val in enumerate(smoothed):
        #    if (val < threshold[i]) and not min_found:
        #        wf_re = head["xs"][i]
        #        wf_re_y = val
        #        new_data_start += i # the new window will always be smaller
        #        min_found = True
        #    if (val > threshold[i]) and min_found:
        #        wf_fe = head["xs"][i]
        #        wf_fe_y = val
        #        new_data_stop -= (len(avg) -i)
        #        break

        #print ("Found le {} and fe {}".format(wf_re,wf_fe))
        #data_re = (wf_re - head["xzero"])/head["xincr"]
        #data_fe = (wf_fe - head["xzero"])/head["xincr"]
        #print("Translated to data points* {} {}".format(new_data_start,new_data_stop))

        #new_data_min = data_re - ((1e-9)*leading/head["xincr"])
        #new_data_max = data_fe + ((1e-9)*trailing/head["xincr"])

        #new_data_start -= ((1e-9)*leading/head["xincr"])
        #new_data_stop += ((1e-9)*trailing/head["xincr"])

        print ("Will set new acquisition window to {} {}"\
               .format(new_data_start, new_data_stop))
        self.scope.data_start = new_data_start
        self.scope.data_stop = new_data_stop

        fig = plotting.plot_waveform(head,avg)
        #fig = plotting.plot_waveform(head, threshold, fig=fig)
        #fig = plotting.plot_waveform(head, smoothed, fig=fig)

        ax = fig.gca()
        #print (wf_re)
        #print (wf_re_y)
        #print (wf_fe)
        #print (wf_fe_y)
        #ax.scatter([wf_re*1e9], [wf_re_y*1e3], color="k", marker="v", label="RE, FE")#, size=15)
        #ax.scatter([wf_fe*1e9], [wf_fe_y*1e3], color="k", marker="v", label="RE, FE")  # , size=15)

        #plot_data_min = wf_re - ((1e-9) * leading)
        #plot_data_max = wf_fe + ((1e-9) * trailing)

        ax.vlines(head["xs"][new_data_start - data_start]*1e9, ax.get_ylim()[0], ax.get_ylim()[1])
        ax.vlines(head["xs"][new_data_stop - data_start]*1e9, ax.get_ylim()[0], ax.get_ylim()[1])
        p.show()

    def get_average_waveform(self,n=10):
        """
        Acquire some waveforms and take the average

        Keyword Args.
            n (int): number of waveforms to average over

        Returns:
            tuple(np.array). xs, ys

        """
        # Take ten waveforms
        wf = self.make_n_acquisitions(n)
        len_wf = [len(w) for w in wf]
        wf = [w for w in wf if len(w) == min(len_wf)]
        avg = reduce(lambda x, y: x+y, wf)/n
        return self.wf_header["xs"], avg

    def set_acquisiton_window(self,start,stop):
        """
        Set the acquisiton window in scope units

        Args:
            start:
            stop:

        Returns:
            None
        """

        self.scope.data_start = int(start)
        self.scope.data_stop = int(stop)

    def show_waveforms(self):
        """
        Demonstration function: Will use pylab show to
        plot some acquired waveforms

        Returns:
            None
        """

        head = self.scope.get_wf_header()
        wf = self.make_n_acquisitions(5)
        for i in range(len(wf)):
            try:
                plotting.plot_waveform(head,wf[i])
            except Exception as e:
                print (e)

        p.show()





