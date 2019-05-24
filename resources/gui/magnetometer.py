"""
Read out a Lutron instrument via USB serial connection.
"""

import serial as s
import numpy as np
import time
import pylab as p
import seaborn as sb
import argparse
import abc

from six import with_metaclass

try:
    import zmq
except ImportError:
    print("Can not import zero MQ")

from . import get_logger

# example word... \r\x0241B20200000052\

# Connection settings:
# Baud rate 9600 
# Parity  No parity
# Data bit no. 8 Data bits
# Stop bit  1 Stop 
#

def save_execute(func):
    """
    Calls func with args, kwargs and 
    returns nan when func raises a value error
    
    FIXME: treatment of the errors!
    """
    def wrap_f(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except ValueError:
            return np.nan
    return wrap_f


def identify_instrument(instrument):
    """
    Find out if the instrument is a Gaussmeter or a Thermometer

    Args:
        instrument:

    Returns:
        str
    """
    some_data = ""
    while not len(some_data) == 15:
        time.sleep(2)
        some_data = instrument.meter.read_all()
        some_data = some_data.split(b"\r")[0]
    unit = some_data[3:5].decode()
    if unit in ["B3", "B2"]:
        return "Magnetometer"
    else:
        return "Thermometer"

#@save_execute

class LutronAbstractBaseInstrumentProxy(with_metaclass(abc.ABCMeta,object), object):
    """
    Proxy to access instruments over the network

    """

    @abc.abstractproperty
    def pattern(self):
        """
        Defines how to find the data out of the publish stream

        Returns:
            str
        """
        return None

    def __init__(self, ip, port, pull_interval=10):
        """
        Subscribe to a Lutron instrument which publishes data with a ZMQ publish socket.
        Lutron instrument needs to publish at ip and port

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
        self.socket.connect("tcp://{}:{}".format(ip, port))
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



class LutronAbstractBaseInstrument(with_metaclass(abc.ABCMeta,object), object):
    """
    Common abstract base class for all Lutron instruments

    """

    def __init__(self, device="/dev/ttyUSB0", loglevel=20,\
                 publish=False, port=9876):
        """
        Constructor needs read and write access to
        the port

        Keyword Args:
            port (str): The virtual serial connection on a UNIX system
            loglevel (int): 10: debug, 20: info, 30: warnlng....
            publish (bool): publish data on port
            port (int): use this port if publish = True
        """
        self.meter = s.Serial(device) # default settings are ok
        self.logger = get_logger(loglevel)
        self.logger.debug("Meter initialized")
        self.publish = publish
        self.port = port
        self._socket = None
        self.axis_label = None

    def _setup_port(self):
        """
        Setup the port for publishing

        Returns:
            None
        """
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        self._socket.bind("tcp://0.0.0.0:%s" % int(self.port))
        return

    def measure_continously(self, npoints, interval):
        """
        Make a measurement with npoints each interval seconds

        Args:
            npoints (int): number of measurement points
            interval (int): make a measurement each interval seconds

        Keyword Args:
            silent (bool): Suppress output

        """

        for n in range(npoints):
            field = self.measure(measurement_time=interval)
            yield n * interval, field

    @property
    def unit(self):
        """
        Figure out the units of the meter
        """
        time.sleep(1)
        some_data = self.meter.read_all()
        some_data = some_data.split(b"\r")[0]
        unit = None
        while unit is None:
            try:
                unit = self.get_unit(some_data)
            except ValueError:
                self.logger.debug("Can not get unit from {}, trying again...".format(some_data))
                time.sleep(2)
                some_data = self.meter.read_all()
                some_data = some_data.split(b"\r")[0]
        self.logger.info("All data will be in {}".format(unit))
        return unit

    @abc.abstractstaticmethod
    def get_unit(self, data):
        """
        Get the unit of the measurment

        Returns:
            str
        """
        return None

    @abc.abstractclassmethod
    def decode_output(self, data):
        """
        Decode the bytestring with
        hex numbers to the field
        information

        Args:
            data (bytes): raw output from serial port

        """

        return data

    @staticmethod
    def _decode_fields(data):
        """
        Decode the individual fields in the outp-ut

        Returns:
            np.ndarray
        """
        if not len(data) == 15:
            return np.nan
        polarity = data[5:6].decode()
        decimal_point = data[6:7].decode()
        mg_data = data[7:15].decode()
        polarity = int(polarity)
        decimal_point = int(decimal_point)
        if polarity:
            polarity = -1
        else:
            polarity = 1
        if decimal_point:
            index = -1 * int(decimal_point)
            mg_data = mg_data[:index] + "." + mg_data[index:]
        mg_data = polarity * float(mg_data)
        return mg_data

    def measure(self, measurement_time=2):
        """
        Measure a single point

        Keyword Args:
            measurement_time (int): average the values over the measurement time in seconds

        """
        time.sleep(measurement_time) # give the meter time to acquire some data
        data = self.meter.read_all()
        try:
            unit = self.get_unit(data.split(b"\r")[0])
        except ValueError:
            unit = "--"
        field = self.decode_output(data)
        field = field.mean()
        if self.publish and (self._socket is None):
            self._setup_port()
        if self.publish:
            self._socket.send_string("{} {}; {}".format("GU3001D", field, unit))
        return field

class GaussMeterProxy(LutronAbstractBaseInstrumentProxy, object):
    """
    Access gaussmeter data via the network

    """

    @property
    def pattern(self):
        pattern = re.compile("GU3001D\s(?P<value>[0-9-.]*);\s+(?P<unit>(mG)|(uT))")
        return pattern


class GaussMeterGU3001D(LutronAbstractBaseInstrument):
    """
    The named instrument
    """

    def __init__(self, *args, **kwargs):
        LutronAbstractBaseInstrument.__init__(self, *args, **kwargs)
        self.axis_label = "Magnetic field"

    @staticmethod
    def get_unit(data):
        """
        Get the unit from a magnetometer word
        """
        if not len(data) == 15:
            raise ValueError("Data corrupted!")
        unit = data[3:5].decode()
        if unit == "B3":
            unit = "mG"
        elif unit == "B2":
            unit = "uT"
        else:
            raise ValueError("Unit not understood {}".format(unit))
        return unit



    def decode_output(self, data):
        """
        Decode the bytestring with
        hex numbers to the field
        information

        Args:
            data (bytes): raw output from serial port

        """
        data = data.split(b"\r")
        data = [self.decode_fields(word) for word in data]
        data = np.array(data)
        data = data[np.isfinite(data)]
        return data

    def decode_fields(self, data):
        """
        Give meaning to the fields of a 16 byte
        word returned by the meter. Words are
        separated by \r

        Args:
            data (bytes): A single word of length 16

        """
        #### from the manual...
        # http://www.sunwe.com.tw/lutron/GU-3001eop.pdf
        # The 16 digits data stream will be displayed in the
        # following  format :
        # D15 D14 D13 D12 D11 D10 D9 D8 D7 D6 D5 D4 D3 D2 D1 D0
        # Each digit indicates the following status :
        # D15  Start Word = 02
        # D14  4
        # D13  When send the upper display data = 1
        #      When send the lower display data = 2
        # D12 & Annunciator for Display
        # D11   mG = B3
        #       uT = B2
        # D10 Polarity
        #       0 = Positive
        #       1 = Negative
        # D9  Decimal Point(DP), position from right to the
        #       left, 0 = No DP, 1= 1 DP, 2 = 2 DP, 3 = 3 DP
        # D8 to D1  Display reading, D8 = MSD, D1 = LSD
        #           For example :
        #           If the display reading is 1234, then D8 to
        #           D1 is : 00001234
        #          D0 End Word = 0D
        return self._decode_fields(data)


class ThermometerTM947SD(LutronAbstractBaseInstrument):
    """
    A 4 channel thermometer

    """
    def __init__(self, *args, **kwargs):
        LutronAbstractBaseInstrument.__init__(self, *args, **kwargs)
        self.active_channel = None
        self.axis_label = "Temperature "

    def select_channel(self, channel):
        """
        Select one of the channels 1-4

        Args:
            channel (int): select this channel

        Returns:
            None
        """
        self.active_channel = channel

    @staticmethod
    def decode_channel(data):
        """
        Find out which channel has the data

        Returns:

        """
        try:
            return int(data[2:3].decode())
        except ValueError:
            return None

    def decode_fields(self, data):
        """
        Give meaning to the fields of a 16 byte
        word returned by the meter. Words are
        separated by \r

        Args:
            data (bytes): A single word of length 16

        """
        #### from the manual...
        # http://www.sunwe.com.tw/lutron/GU-3001eop.pdf
        # The 16 digits data stream will be displayed in the
        # following  format :
        # D15 D14 D13 D12 D11 D10 D9 D8 D7 D6 D5 D4 D3 D2 D1 D0
        # Each digit indicates the following status :
        # D15  Start Word = 02
        # D14  4
        # D13  When send the T1 display data = 1
        #      When send the T2 display data = 2
        #      When send the T2 display data = 3
        #      When send the T2 display data = 4
        # D12 & Annunciator for Display
        # D11   Celsius = 01
        #       Fahrnen = 02
        # D10 Polarity
        #       0 = Positive
        #       1 = Negative
        # D9  Decimal Point(DP), position from right to the
        #       left, 0 = No DP, 1= 1 DP, 2 = 2 DP, 3 = 3 DP
        # D8 to D1  Display reading, D8 = MSD, D1 = LSD
        #           For example :
        #           If the display reading is 1234, then D8 to
        #           D1 is : 00001234
        #          D0 End Word = 0D
        return self._decode_fields(data)

    def decode_output(self, data):
        """
        Decode the bytestring with
        hex numbers to the field
        information

        Args:
           data (bytes): raw output from serial port

        """
        data = data.split(b"\r")
        data = [self.decode_fields(word) for word in data if self.decode_channel(word) == self.active_channel]
        data = np.array(data)
        data = data[np.isfinite(data)]
        return data

    @staticmethod
    def get_unit(data):
        """
        Get the unit from a magnetometer word
        """
        if not len(data) == 15:
            raise ValueError("Data corrupted!")
        unit = data[3:5].decode()
        if unit == "01":
            unit = "C"
        elif unit == "02":
            unit = "F"
        else:
            raise ValueError("Unit not understood {}".format(unit))
        return unit
 
