"""
Package to read out TektronixDPO4104B oscilloscope 
"""

__version__ = "0.0.16"
__all__ = ["instruments", "controllers", "scpi", "plotting", "tools"]

# global loglevel for this module
LOGLEVEL = 20

def set_loglevel(level):
    """
    Set the global loglevel for this module.

    Args:
        level (int): Severity. 10 = debug, 20 = info, 30 = warn

    Returns:
        None
    """
    LOGLEVEL = level
    return None

def _hook():
    pass
