"""
Talk to Textronic scope via sockets
"""

import argparse
import socket

from . import commands as cmd

import time
import numpy as np
import vxi11

import logging

from copy import copy

# abbreviations
dec = cmd.decode
enc = cmd.encode
aarg = cmd.add_arg
q = cmd.query


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
    return property(lambda self: self.send(q(command)),\
                    lambda self, value: self.set(aarg(command, value)))


class TektronixDPO4104B(object):
    """
    Oscilloscope of type DPO4104B manufactured by Tektronix
    """

    # constants used by the socket connection
    WAITTIME = .25
    SOCK_TIMEOUT = 5
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

    @property
    def get_triggerrate(self):
        """
        The rate the scope is triggering. This number is provided
        by the scope. Most times it is nan though...

        Returns:
            float
        """
        trg_rate = self.send(cmd.TRG_RATEQ)
        trg_rate = float(trg_rate)
        # from the osci docs
        # the IEEE Not A Number (NaN = 99.10E+36)
        if trg_rate > 1e35:
            trg_rate = np.nan
        return trg_rate

    def __init__(self,ip="169.254.68.19",port=4000):
        """
        Connect to the scope via its socket server

        Args:
            ip (str): ip of the scope
            port (int): port the scope is listening at
        """
        self.ip = ip
        self.port = port
        self.connect_trials = 0
        self.wf_buff_header = None # store a waveform header in case they are all the same
        #self._osock = socket.create_connection((ip,port),self.SOCK_TIMEOUT)
        self.instrument = vxi11.Instrument(ip)

    #def __del__(self):
    #    """
    #    Close the socket
    #    """
    #    self._osock.close()

    def reopen_socket(self):
        """
        Close and reopoen the socket after a timeout

        Returns:
            None
        """
        #self._osock = socket.create_connection((self.ip, self.port), self.SOCK_TIMEOUT)
        self.instrument = vxi11.Instrument(self.ip)

    def send(self,command,buffsize=2**16):
        """
        Send command to the scope. Raises socket.timeout error if
        it had failed too often

        Args:
            command (str): command to be sent to the 
                           scope

        """
        if self.connect_trials == self.MAXTRIALS:
            self.connect_trials = 0
            raise socket.timeout

        if self.verbose: print ("Sending {}".format(enc(command)))
        try:
            response = self.instrument.ask(command)
        except Exception as e:
            self.reopen_socket()
            response = self.instrument.ask(command)
            self.connect_trials += 1

        return response
        #self._osock.send(enc(command))
        #time.sleep(self.WAITTIME)
        #try:
        #    response = self._osock.recv(buffsize)
        #except socket.timeout:
        #    self.connect_trials += 1
        #    self.reopen_socket()
        #    time.sleep(self.WAITTIME)
        #    response = self.send(command)

        return dec(response)

    def set(self, command):
        """
        Send a command bur return no response

        Args:
            command (str): command to be send to the scope

        Returns:
            None
        """
        if self.verbose: print("Sending {}".format(enc(command)))
        self.instrument.write(command)
        #self._osock.send(enc(command))

    def ping(self):
        """
        Check if oscilloscope is connected
        """

        ping = self.send(cmd.WHOAMI)
        print (ping)
        return True if ping else False

    def get_histogram(self):
        """
        Return a histogram which might be recorded
        by the scope
        """
        start = self.histstart
        end = self.histend
        bincontent = self.send(cmd.HISTDATA)
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

    def get_wf_header(self):
        """
        Get some meta information about the *next incoming wavefrm*

        Returns:
            dict
        """
        header = self.send(cmd.WF_HEADER)
        header = cmd.parse_custom_wf_header(header)
        self.wf_buff_header = header
        #header["xs"] = np.ones(len(header["npoints"]))*header["xzero"]
        xs = np.ones(header["npoints"])*header["xzero"]

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
        waveform = self.send(cmd.CURVE)
        if single_acquisition: self.acquire = cmd.OFF
        #self.get_wf_header()
        waveform = np.array([float(k) for k in waveform.split(",")])
        if self.wf_buff_header is None:
           self.get_wf_header()

        # from the docs
        # Value in YUNit units = ((curve_in_dl - YOFf) * YMUlt) + YZEro
        waveform = (waveform - (np.ones(len(waveform))*self.wf_buff_header["yoff"]))\
                   * (np.ones(len(waveform))*self.wf_buff_header["ymult"])\
                   + (np.ones(len(waveform))*self.wf_buff_header["yzero"])

        return waveform

#################################################3


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port",help="The port the oscilloscope is listening",type=int,default=4000)
    parser.add_argument("-i","--ip",help="The ip adress of the oscilloscope",type=str,default="169.254.14.30")

    args = parser.parse_args()

    scope = TektronixDPO4104B(args.ip,args.port)
    #scope._osock.send("*IDN?\r\n".encode("utf8"))
    #print ("querying")
    #print(scope._osock.recv(2**32))
    #scope._osock.send(u"*IDN?\r\n".encode("latin-1"))
    #print(scope._osock.recv(2**32))
    
    scope.ping()
    #print (scope.query(cmd.SOURCE))
    #scope.query(cmd.SOURCE)
    #print ( scope.get_histogram())
    scope.get_waveform()
