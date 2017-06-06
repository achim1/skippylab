from . import oscilloscopes

class Agilent33220A(oscilloscopes.AbstractBaseOscilloscope):

    def __init__(self, ip="10.25.21.168"):
        """
        An Agilent function generator

        Args:
            ip: The port where this instrument is listening on
        """
        oscilloscopes.AbstractBaseOscilloscope.__init__(self, ip=ip)
        