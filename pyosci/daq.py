"""
Use the scope as a DAQ

"""
from . import osci
from . import tools
from . import commands as cmd
from . import plotting
from . import logging as daqlog

logger = daqlog.get_logger(20)

import time
import re
import numpy as np
import pylab as p
try:
    import zmq
except ImportError:
    logger.warning("No zmq available!")


from socket import timeout as TimeoutError
from datetime import datetime


bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    logger.warning("No pyprind available")

try:
    from functools import reduce
except ImportError:
    logger.warning("Can not import functools, this might be python 2.7?")

# helpers


class GaussMeterProxy(object):
    """
    Access gaussmeter data via the network

    """

    def __init__(self, ip, port, pull_interval=10):
        """
        Subscribe to a Gaussmeter which publishes data with a ZMQ publish socket.
        Gaussmeter needs to publish at ip and port

        Args:
            ip (str): ip of gaussmeter (running goldschmidt instance)
            port (int): port where guassmeter is publishing

        Keyword Args:
            pull_interval (int): pull data new data every pull_interval seconds, use
            buffered value in between
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'GU3001D')
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect("tcp://{}:{}".format(ip,port))
        self.pattern = re.compile("GU3001D\s(?P<value>[0-9-.]*);\s+(?P<unit>(mG)|(uT))")
        self.pull_interval = pull_interval
        self.last_pull = None
        self.buffer = self._pull_new()

    def _pull_new(self):
        """
        Acquire new data

        Returns:
            dict
        """
        self.last_pull = time.monotonic()
        try:
            self.buffer = self.pattern.search(self.socket.recv().decode()).groupdict()
        except Exception as e:
            logger.debug("Problem acquiring data {}".format(e))
        self.buffer["value"] = float(self.buffer["value"])
        return self.buffer

    def pull(self):
        """
        Get new data or returns the data in the buffer dependent on self.pull_interval

        Returns:
            dict
        """
        if time.monotonic() - self.last_pull > self.pull_interval:
            return self._pull_new()
        else:
            return self.buffer



class Event(object):
    """
    DAQ will return events when triggered.
    """

    def __init__(self, use_datetime=False):
        """
        Keyword Args:
            use_datetime (bool): if True, give timestamp with datetime.datetime
        """
        self.use_datetime = use_datetime
        self.data = dict()
        self.timestamp = None

    def timestamp_it(self):
        """
        Give it a timestamp! Time in seconds

        Returns:
            None
        """
        if self.use_datetime:
            self.timestamp = datetime.now()
        else:
            self.timestamp = time.monotonic()


class DAQ(object):
    """
    A virtual DAQ using an oscilloscope
    """

    def __init__(self):
        """
        Initialize a new collector for instrument data
        """
        self.channels = dict()

    def register_instrument(self, instrument, label="instrument"):
        """
        Register an instrument and assign a channel to it. Instruments must have a pull()
        method which allows to pull data from them at a certain event.

        Args:
            instrument (ducktype): needs to be configured already and must have a pull() method
            channel_name (int): identify the instrument under this registered channel

        Returns:
            None
        """
        assert label not in self.channels.keys(),\
            "Instrument with label {} already registered! Chose a different label".format(label)

        self.channels[label] = instrument

    def acquire(self):
        """
        Go through the instrument list and trigger their pull methods to build an event

        Returns:
            pyosci.Event
        """
        event = Event()
        for key in self.channels.keys():
            event.data[key] = self.channels[key].pull()
        return event

    def acquire_n_events(self, n_events, trigger_hook=lambda x:x):
        """
        Continuous acquisition. Acquires n events. Yields events. Use trigger hook to define a
        function to decide when data is returned.

        Args:
            n_events (int): Number of events to acquire
            trigger_hook (callable): Trigger condition

        Yields:
            Event
        """
        if bar_available:
            bar = pyprind.ProgBar(n_events, track_time=True, title='Acquiring waveforms...')

        for __ in range(n_events):
            trigger_hook()
            if bar_available:
                bar.update()
            yield self.acquire()







