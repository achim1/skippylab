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


from .controllers import DirectUSBController,\
                         GPIOController,\
                         ZMQController,\
                         PrologixUsbGPIBController,\
                         NI_GPIB_USB,\
                         SimpleSocketController,\
                         TelnetController
