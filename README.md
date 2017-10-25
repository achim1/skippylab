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


Supported oscilloscopes 
----------------------------------

* Tektronix DPO4104B

* Rhode&Schwarz RTO1044


Supported function generators
----------------------------------------

* Agilent 33220A (via GPIB)


Supported power supplies
---------------------------------------

* Keysight E3631A




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


