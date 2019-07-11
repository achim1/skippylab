"""
Provides a number of controllers to connect instruments via an usb emulated
serial port. This can mean to hook up the instrument directly to the USB port
or use a GPIB-USB interface
"""

import abc
import zmq
import serial

from time import sleep

from six import with_metaclass

from . import get_logger

class AbstractBaseController(with_metaclass(abc.ABCMeta,object),object):
    """
    Defines the minimal interface of a controller
    """

    def __init__(self):
        pass
     
    @abc.abstractmethod
    def write(self):
        pass

    @abc.abstractmethod
    def query(self, question):
        pass
    
    def read(self):
        raise NotImplementedError("Plain read is not implemented for this instrument, only 'query' and 'write'")


class DirectUSBController(AbstractBaseController):
    """
    Use when connecting an instrument directly to the USB prot, without any 
    intermediate interface
    """
    def __init__(self, device="/dev/ttyUSB0", baudrate=9600, bytesize=8,
                 parity='N', stopbits=1, timeout=None,
                 xonxoff=False, rtscts=False, write_timeout=None,
                 dsrdtr=False, inter_byte_timeout=None, exclusive=None):
        """
        Open a new serial connection via an USB connection

        Keyword Args:
            device (str): The device port where the instrument is listening (usually /dev/ttyUSB0)
            baudrage (int) : passed thru to serial
            bytesize (int) : passed thru to serial
            parity   (str) : passed thru to serial
            stopbits (int) : passed thru to serial
            timeout  (int) : passed thru to serial
            xonxoff  (bool) : passed thru to serial
            rtscts   (int) : passed thru to serial
            write_timeout  (int) : passed thru to serial
            dsrdtr   (bool)
            inter_byte_timeout   : passed thru to serial
            exclusive            : passed thru to serial
        """
        self.conn = serial.Serial(port=device, baudrate=baudrate, bytesize=bytesize,
                                  parity=parity, stopbits=stopbits, timeout=timeout,
                                  xonxoff=xonxoff, rtscts=rtscts, write_timeout=write_timeout,
                                  dsrdtr=dsrdtr, inter_byte_timeout=inter_byte_timeout,
                                  exclusive=exclusive)


    def query(self,command):
        self.conn.write("{}\r\n".format(command).encode())
        sleep(0.3)
        resp = self.conn.read_all()
        return resp.decode().rstrip("\n")

    def write(self,command):
         self.conn.write("{}\r\n".format(command).encode())

    def read(self):
        return self.conn.read_all().decode()


class PrologixUsbGPIBController(AbstractBaseController):

    def __init__(self, port='/dev/ttyUSB0', gpib_adress=6, stopbits=2, set_auto_mode=True):
        self.conn = serial.Serial(port=port, stopbits=stopbits)    
        self.conn.write(f"++addr {gpib_adress}\n".encode())
        self.conn.write("++eos 0\n".encode())
        if set_auto_mode:
            self.conn.write("++auto 1\n".encode())


    def query(self, command):
        self.conn.write(f"{command}\n\n".encode())
        sleep(0.3) 
        resp = self.conn.read_all()
        return resp.decode().rstrip("\n")

    def write(self, command):
        self.conn.write(f"{command}\n\n".encode())
        return None


try:
    import visa
except ImportError:
    print ("Can not use NI_GPIB_USB controller, need to install visa library. Search pypi or github")

class NI_GPIB_USB(AbstractBaseController):

    def __init__(self,gpib_adress=6, port=9999, publish=False):
        resource_manager = visa.ResourceManager()
        print("Found the following visa resources : {}".format(resource_manager.list_resources()))
        chamber_found = False
        try:
            self.resource = resource_manager.open_resource(resource_manager.list_resources()[0])
            resource_found = True
        except IndexError:
            print("Can not find any visa resources!")
            self.resource = None


class GPIOController(AbstractBaseController):
    """
    Controls readout from GPIO pins of raspberry pi
    """

    def __init__(self, gpio_pins=None,
                 data_getter=lambda : None,
                 data_setter=lambda x : None):
        """
        Keyword Args:
            gpio_pins (iterable of ints) : list of gpio pins to use
            data_getter (callable)       : callback to retrieve data
            data_setter (callable)       : send commands to GPIO pins
        """
        self.pins = gpio_pins
        self.data_getter = data_getter

    def read(self):
        return self.data_getter()

    def write(self, command):
        self.data_setter(command)

    def query(self):
        raise NotImplementedError("Usually GPIO interfaces do not do queries, but if yours does, please override this method")




