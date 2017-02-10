"""
Prepare logging functionality for the module
"""

import sys
import logging
import os

LOGFORMAT = '[%(asctime)s] %(levelname)s: %(module)s(%(lineno)d):   %(message)s'


def get_logger(loglevel,logfile=None):
    """
    A root logger with a formatted output logging to stdout and a file

    Args:
        loglevel (int): 10,20,30,... the higher the less logging
        logfile (str): write logging to this file as well as stdout
    Returns:
        logging.logger
    """

    def exception_handler(exctype, value, tb):
        logger.critical("Uncaught exception", exc_info=(exctype, value, tb))

    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    ch = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            ch = h
            break

    if ch is None:
        ch = logging.StreamHandler()

    ch.setLevel(loglevel)
    formatter = logging.Formatter(LOGFORMAT)
    ch.setFormatter(formatter)

    if logfile is not None:
        today = datetime.now()
        today = today.strftime("%Y-%m-%d_%H-%M")
        logend = ".log"
        if logfile.endswith(".log"):
            logfile.replace(".log",today+logend)
        else:
            logfile += (today + logend)
        logfilecount = 1

        # find a file name which does not exist yet
        while os.path.exists(logfile):
            logfile = logfile.replace("." + str(logfilecount -1),"")
            logfile = logfile +"." + str(logfilecount)
            logfilecount += 1
            if logfilecount >= 60:
                raise SystemError("More than 1 logfile per second, this is insane.. aborting")

        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(loglevel)

    logger.addHandler(ch)
    if logfile is not None: logger.addHandler(fh)
    sys.excepthook = exception_handler
    logger.propagate = False
    return logger
