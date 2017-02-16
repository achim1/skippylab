"""
Package to read out TektronixDPO4104B oscilloscope 
"""

__version__ = "0.0.13"
__all__ = ["daq", "osci", "plotting", "tools", "fit"]

import appdirs as _appdirs
import shutil as _shutil
import os as _os

from matplotlib import get_configdir as mpl_configdir

from . import logging

_appdir = _os.path.split(__file__)[0]
_appname = _os.path.split(_appdir)[1]

# the config files
STYLE_BASEFILE_STD = _os.path.join(_appdir,"pyoscipresent.mplstyle")
STYLE_BASEFILE_PRS = _os.path.join(_appdir,"pyoscidefault.mplstyle")

logger = logging.get_logger(20)

def get_configdir():
    """
    Definges a configdir for this package under $HOME/.pyevsel
    """
    config_dir = _appdirs.user_config_dir(_appname)
    if not _os.path.exists(config_dir):
        _os.mkdir(config_dir)
    return config_dir

def install_styles(style_default=STYLE_BASEFILE_STD, \
                   style_present=STYLE_BASEFILE_PRS):
    """
    Sets up style files

    Keyword Args:
        style_default (str): location of style file to use by defautl
        style_present (str): location of style file used for presentations
        plots_config (str): configureation file for plots
        patternfile (str): location of patternfile with file patterns to search and read
    """

    logger.info("Installing styles...")
    cfgdir = get_configdir()
    mpl_styledir = _os.path.join(mpl_configdir(),"stylelib")
    for f in style_default, style_present:
        assert _os.path.exists(f), "STYLEFILE {} missing... indicates a problem with some paths or corrupt packege.\
                                    Check source code location".format(f)
        _shutil.copy(f,mpl_styledir)

install_styles()
