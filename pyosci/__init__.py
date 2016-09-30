"""
Package to read out TektronixDPO4104B oscilloscope 
"""

__version__ = "0.0.1"

import appdirs
import shutil
import os

from matplotlib import get_configdir as mpl_configdir

_appdir = os.path.split(__file__)[0]
_appname = os.path.split(_appdir)[1]

# the config files
STYLE_BASEFILE_STD = os.path.join(_appdir,"pyoscipresent.mplstyle")
STYLE_BASEFILE_PRS = os.path.join(_appdir,"pyoscidefault.mplstyle")


def get_configdir():
    """
    Definges a configdir for this package under $HOME/.pyevsel
    """
    config_dir = appdirs.user_config_dir(_appname)
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
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

    print ("Installing styles...")
    cfgdir = get_configdir()
    mpl_styledir = os.path.join(mpl_configdir(),"stylelib")
    for f in style_default, style_present:
        assert os.path.exists(f), "STYLEFILE {} missing... indicates a problem with some paths or corrupt packege. Check source code location".format(f)
        shutil.copy(f,mpl_styledir)

install_styles()