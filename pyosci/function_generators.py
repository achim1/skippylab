from . import osci

class Agilent33220A(osci.AbstractBaseOscilloscope):

    def __init__(self, ip="10.25.21.168"):
        """
        An Agilent function generator

        Args:
            ip: The port where this instrument is listening on
        """
        osci.AbstractBaseOscilloscope.__init__(self, ip=ip)
        