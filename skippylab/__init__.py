"""
Package to read out TektronixDPO4104B oscilloscope 
"""

__version__ = "0.0.15"
__all__ = ["instruments", "scpi", "plotting", "tools"]

from hepbasestack import logger
LOGLEVEL = logger.LOGLEVEL
Logger = logger.Logger

from . import instruments, scpi, plotting, tools

# easy access
TektronixDPO4104B = instruments.TektronixDPO4104B

def set_loglevel(level):
    """
    Set the loglevel, 10 = debug, 20 = info, 30 = warn
    """
    logger.LOGLEVEL = level
    return

def _hook():
    pass
