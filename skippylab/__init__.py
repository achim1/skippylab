"""
Package to read out TektronixDPO4104B oscilloscope 
"""

__version__ = "0.0.15"
__all__ = ["instruments", "scpi", "plotting", "tools"]

import appdirs as _appdirs
import shutil as _shutil
import os as _os

from . import instruments, scpi, plotting, tools

from . import loggers
from .loggers import get_logger

# easy access
TektronixDPO4104B = instruments.TektronixDPO4104B


logger = loggers.get_logger(20)


def _hook():
    pass
