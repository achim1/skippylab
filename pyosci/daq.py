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

    def acquire(self, *pullargs, **pullkwargs):
        """
        Go through the instrument list and trigger their pull methods to build an event

        Keyword Args:
            **pullkwargs (dict): will be passed on the individual pull methods

        Returns:
            pyosci.Event
        """
        event = Event()
        for key in self.channels.keys():
            event.data[key] = self.channels[key].pull(*pullargs, **pullkwargs)
        return event

    def acquire_n_events(self, n_events, trigger_hook=lambda x:x,\
                         trigger_hook_args=(None,), pull_args=(), pull_kwargs={}):
        """
        Continuous acquisition. Acquires n events. Yields events. Use trigger hook to define a
        function to decide when data is returned.

        Args:
            n_events (int): Number of events to acquire
            trigger_hook (callable): Trigger condition
            trigger_hook_args (tuple): Arguments for the trigger condition
        Yields:
            Event
        """
        if bar_available:
            bar = pyprind.ProgBar(n_events, track_time=True, title='Acquiring waveforms...')

        for __ in range(n_events):
            trigger_hook(*trigger_hook_args)
            if bar_available:
                bar.update()
            yield self.acquire(*pull_args, **pull_kwargs)







