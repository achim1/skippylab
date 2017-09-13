[![Code Issues](https://www.quantifiedcode.com/api/v1/project/bd0c238d3dd3406d8dc2d4a456a862e1/badge.svg)](https://www.quantifiedcode.com/app/project/bd0c238d3dd3406d8dc2d4a456a862e1)

SKippylab - work with SCPI instruments
========================================================================


Rationale
--------------



Use a vxi11 capable scope (most of the new ones with a LAN port and visa compatibility support that protocoll) as a data acquisition system.
The package provides a oscilloscope independent DAQ which can be used for scripting the data taking.

Requirements
--------------

For the software to work, the machine this software is installed on needs to be in the same network as the scope and its ip adress must be known.
*tested only for python 3.5 so far*



Supported oscilloscopes 
----------------------------------

* TektronixDPO4104B

* Rhode&SchwarzRTO

Finding out the scope's IP adress (Tektronix)
-------------------------------------------------

* Press the utility button

* Select io with the mainpurpose dial (upper left one)

* LAN settings


Installation 
--------------

`pip3 install pyosci`



Using the software
---------------------

An example is given by the "measurement.py" script which can be found in pyosci/resources. Acquisition of
waveforms can be performed e.g. like this.


```
import pyosci.osci as osci
import pyosci.daq as daq
import pyosci.tools as tools

# initialize with scope IP adress and port
odaq = daq.DAQ(osci.TektronixDPO4104B("169.254.67.106"))

# set an acquisiton window -20 +100 ns around peak in waveform
odaq.set_feature_acquisition_window(20,100) 

# display some waveforms on the screen
odaq.show_waveforms()

# header stores information about x and y ticks
head = odaq.scope.get_wf_header()

# take actual data (e.g. 50000 waveforms) and save it to a 
# numpy compatible file 
wf = odaq.make_n_acquisitions(50000)
tools.save_waveform(head, wf, "measurement_ctrl_vlt")


```

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


