[![Docs](https://readthedocs.org/projects/skippylab/badge/?version=latest)](http://skippylab.readthedocs.io/en/latest/?badge=latest)


SKippylab - work with SCPI instruments
========================================================================


Rationale
--------------

The [![scpi protocoll](https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments)](https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments) allows for a fairly standardized way to control electronic equipment via the network.

This software is intended for the use in a lab where instruments like function generators, oscilloscopes need to work together in a coherent way.

To support instruments which do not have an ethernet interface but are equipped with a [![GPIB interface](https://en.wikipedia.org/wiki/IEEE-488)](https://en.wikipedia.org/wiki/IEEE-488), the software relies on [![Prologix GPIB controllers](http://prologix.biz/gpib-ethernet-controller.html)](http://prologix.biz/gpib-ethernet-controller.html)

Requirements
--------------

* python >= 3.5

* prologix-gpib-ethernet from github/pip: `pip install git+git://github.com/nelsond/prologix-gpib-ethernet.git`


A word on usbtmc
-----------------

Make sure the linux driver module 'usbtmc' is loaded, by checking with 'lsmod | grep tmc'. There is a [![usbtmc driver on github](https://github.com/imrehg/usbtmc.git)](https://github.com/imrehg/usbtmc.git), if it is not available.

Maybe the installation of the usblit can help as well, this is not entirely clear at this point, issue `apt install libusb-1.0-0-dev` on an Ubuntu system 

If the serial number despite the fact pythhon is executed with sudo right for an usbtmc instrument can not be read,[![this can help](https://www.oipapio.com/question-561736)](https://www.oipapio.com/question-561736). 


Visa library implementations
------------------------------

There are a lot of different python packages related to visa installations, some eeven share the same name for the `import` statement! 


Udev rules
------------

Since typically any resource created under `/dev/` will be read-only, `udev` rules are recommended.


Supported instruments
-------------------------


GPIB controllers
--------------------------

Prologix Ethernet/USB (*I highly recommend the prologix controllers, there are cheap and have an open interface*)



Oscilloscopes 
----------------------------------

* Tektronix DPO4104B

* Rhode&Schwarz RTO1044


Function generators
----------------------------------------

* Agilent 33220A (via GPIB)


Power supplies
---------------------------------------

* Keysight E3631A



Patchpannels
------------------

* Cytec



Extending the software
-------------------------

Many commands are of the form getter/setter, like `DATa:STARt` 
can be issued in the form `DATa:STARt?` or `DATa:STARt 1000`.
These getter/setters are modeled by python property objects.

```
def setget(cmd):
    """
    Shortcut to construct property object to wrap getters and setters
    for a number of settings

    Args:
        cmd (str): The command being used to get/set. Get will be a query
        value (str): The value to set

    Returns:
        property object
    """
    return property(lambda self: self.send(q(cmd)),\
                    lambda self, value: self.set(aarg(cmd,value)))
```

This can be used to expand the functionality with such commands easily:

```
class MyNewFancyScope(AbstractBaseOscilloscope):
    """
    Implement functionality for a new scope...
    """

    ...

    mynewcommand = setget("COMMAND:COMMAND")
    
```

And the command will be wrapped as a class attribute.


