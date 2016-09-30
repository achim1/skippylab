pyosci - manage data acquisition with Tectronix DPO 4104B
=============================================================

This package provides an interface to get data out of the 
Tektronix DPO 4104B oscilloscope via a network connection.
The scope's socket server is used for that.
For the software to work, the machine this software is installed on needs to be in the same network as the scope and its ip adress must be known.

Finding out the scope's IP adress
------------------------------------

* Press the utility button

* Select io with the mainpurpose dial (upper left one)

* Lan settings


Installation 
--------------

`pip install pyosci`



Using the software
---------------------

An example is given by the "measurement.py" script which can be found in pyosci/resources. Acquisition of
waveforms can be performed e.g. like this.

```
import pyoaci.daq as daq

# scope's ip address and port
odaq = daq.DAQ("169.254.67.106",4000)

# adjust the time which is waited in between commands to
# your needs
odaq.scope.WAITTIME = 0.1

# this will acquire some waveforms, average them and show a plot with
# a reasonable timewindow for data acquisition
odaq.find_best_acquisition_window(waveforms=20, trailing=100)

# plots some acquired waveforms
odaq.show_waveforms()

# get a number of waveforms from the scope. If return_only_charge is set,
# then they will be integrated directly in order to save memory
wf = odaq.make_n_acquisitions(50000,trials = 100000, extra_timeout=0., \
                              return_only_charge = True)
```

Extending the software
-------------------------

Many commands are of the form getter/setter, like `DATa:STARt` 
can be issued in the form `DATa:STARt?` or `DATa:STARt 1000`.
These getter/setters are modeled by python property objects.

`def setget(cmd):
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
`
This can be used to expand the functionality with such commands easily:

`class TektronixDPO4104B(object):
    """
    Oscilloscope of type DPO4104B manufactured by Tektronix
    """

    ...

    mynewcommand = setget("COMMAND:COMMAND")
    
`
And the command will be wrapped as a class attribute.


