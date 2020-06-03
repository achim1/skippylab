"""
Controllers are used for communication with devices. This can be either 
via ethernet or usb. In case of usb, this then can be through a serial connection, 
or by other means.
The module contains a number of controllers which allow to generically connect
to different instruments.

For ethernet connections:
    ZMQController
    SimpleSocketController
    TelnetController

For connections via a GPIO controller connectoed through USB
    PrologixUsbGPIBController
    NI_GPIB_USB

For direct, serial connection via USB
    DirectUSBController

To take control over GPIO pins on a raspberry pi
    GPIOController
"""
from . import controllers as _controllers
HAS_VISA = _controllers.HAS_VISA

from .controllers import DirectUSBController,\
                         GPIOController,\
                         ZMQController,\
                         PrologixUsbGPIBController,\
                         SimpleSocketController,\
                         TelnetController

if HAS_VISA:
    from .controllers import NI_GPIB_USB

del _controllers

