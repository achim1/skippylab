"""
There are two main types of equipment one can have in the lab:


"""

import zmq

#from ..controllers import DirectUSBController
from ..loggers import get_logger


class AbstractBaseInstrument(object):
    def __init__(self,
                 controller=None,
                 loglevel=20,\
                 publish=False,
                 publish_port=9876):
        """
        Constructor needs read and write access to
        the port

        Keyword Args:
            publish_port (str): The port on which data will be published 
                                in case publish = True
            loglevel (int): 10: debug, 20: info, 30: warnlng....
            publish (bool): publish data on port
            publish_port (int): use this port if publish = True
        """
        self.controller = controller # default settings are ok
        self.logger = get_logger(loglevel)
        self.logger.debug("Instrument initialized")
        self.publish = publish
        self.port = publish_port
        self._socket = None
        self.axis_label = None

    def _setup_port(self):
        """
        Setup the port for publishing

        Returns:
            None
        """
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        self._socket.bind("tcp://0.0.0.0:%s" % int(self.port))
        return

