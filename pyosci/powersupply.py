"""
Connection to power supply unit

"""
import time

from plx_gpib_ethernet import PrologixGPIBEthernet

from . import osci
from . import commands as cmd
from . import logging

bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    pass
    #logger.warning("No pyprind available")

setget = osci.setget
KCmd = cmd.KeysightE3631APowerSupplyCommands

q = cmd.query

class KeysightE3631APowerSupply(object):
    """
    A low volgage power supply with two channels, +6V and +- 25V manufactured
    by Keysight. The power supply does not have an ethernet port, so the
    connection is done via GPIB and a prologix GPIB Ethernet connector
    """
    output = setget(KCmd.OUTPUT)


    def __init__(self, ip="10.25.124.252", gpib_address=5, loglevel=20):
        """
        Connect to the power supply via Prologix GPIB connector

        Keyword Args:
            ip (str): IP adress of the Prologix GPIB connector
            gpib_address (int): The GPIB address of the power supply
                                connected to the Prologix connector
        """
        gpib = PrologixGPIBEthernet(ip)
        gpib.connect()
        gpib.select(gpib_address)
        self.logger = logging.get_logger(loglevel)
        self.instrument = gpib
        self.P6 = KCmd.P6
        self.P25 = KCmd.P25
        self.N25 = KCmd.N25

    def __del__(self):
        self.instrument.close()

    def _query(self, command):
        """
        Send a command

        Args:
            command (str):

        Returns:
            str
        """
        self.logger.debug("Querying {}".format(command))
        return self.instrument.query(command)

    def _set(self, command):
        """
        Send a command bur return no response

        Args:
            command (str): command to be send to the scope

        Returns:
            None
        """

        #FIXME: Make AbstractInstrment class and inherit
        self.logger.debug("Sending {}".format(command))
        self.instrument.write(command)

    def ping(self):
        """
        Check the connection

        Returns:
            str
        """

        return self.instrument.query(cmd.WHOAMI)

    @property
    def error_state(self):
        """
        Read out the error register of the power supply

        Returns:
            str
        """

        error =  self.instrument.query(KCmd.ERROR_STATEQ)
        error = error.split(",")
        err_no = float(error[0])
        return err_no, "".join(error[1:])

    def select_channel(self, channel):
        """
        Select either the +6, +25 or -25V channel

        Args:
            channel (str or int):

        Returns:
            None
        """
        channel_dict = {1: self.P6, 2: self.P25, 3: self.N25}
        if isinstance(channel, int):
            assert channel in [1, 2, 3], "Channel has to be either 0:+6V, 1:+25V or 2: -25V"
            channel = channel_dict[channel]
        elif isinstance(channel, str):
            assert channel in (KCmd.P6, KCmd.P25, KCmd.N25),\
                "Channel has to be in {}".format(KCmd.P6, KCmd.P25, KCmd.N25)
        else:
            raise ValueError("Channel has to be either str or int")

        self._set(KCmd.CHANNEL + " {}".format(channel))
        return channel

    def set_voltage(self, channel, voltage):
        """
        Set the supplied voltage of a channel to the desired value

        Args:
            channel (str or int):
            voltage (float):

        Returns:
            None
        """
        channel = self.select_channel(channel)
        if (channel == self.P6) and (voltage > 6.18):
            raise ValueError("6V Channel does not support {}".format(voltage))
        self._set(KCmd.VOLT + " {}".format(voltage))


    def off(self):
        """
        Cut the power on all channels

        Returns:
            None
        """
        self.logger.info("Disabling power...")
        self.output = KCmd.OFF

    def on(self):
        """
        Enable power on all channels

        Returns:
            None
        """
        self.logger.warning("Enabling power!!")

        # give the user time to hit Ctrl-C to abort
        safety_wait = 5
        if bar_available:
            bar = pyprind.ProgBar(safety_wait, track_time=True, title='Powering on...')

        for i in range(safety_wait):
            time.sleep(1)
            if bar_available:
                bar.update()
            else:
                print ("Powering on in {}".format(safety_wait - i))
        self.output = KCmd.ON

    def measure_current(self, channel):
        """
        Measure current on givven channel

        Args:
            channel (str):

        Returns:
            float
        """
        command = q(KCmd.MEASURE + ":" + KCmd.CURRENT + ":" + KCmd.DC)
        command += " {}".format(channel)
        return float(self._query(command))


