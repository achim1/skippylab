"""
Function generators

"""
import time

from plx_gpib_ethernet import PrologixGPIBEthernet

from . import oscilloscopes as osci
from ..scpi import commands as cmd
from .. import loggers

bar_available = False

try:
    import pyprind
    bar_available = True
except ImportError:
    pass
    #logger.warning("No pyprind available")

setget = osci.setget
#KCmd = cmd.Agilent3322OACommands

q = cmd.query

class Agilent3322OAFunctionGenerator(object):
    """
    """
    #output = setget(KCmd.OUTPUT)

    def __init__(self, ip="10.25.123.111", gpib_address=15, loglevel=20):
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
        self.logger = loggers.get_logger(loglevel)
        self.instrument = gpib
        #self.P6 = KCmd.P6
        #self.P25 = KCmd.P25
        #self.N25 = KCmd.N25

    def __del__(self):
        self.instrument.close()

