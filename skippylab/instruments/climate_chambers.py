#! /usr/bin/env python3
"""
Control SUN EC13 temperature chamber
"""

#from .ni_gpib_usb import NI_GPIB_USB 
from .. controllers import PrologixUsbGPIBController

import numpy as np
import time 

import pylab as p
import hepbasestack as hbs

# visual adaptions for the jupyter notebook
hbs.visual.set_style_default()


try:
    import zmq 
except ImportError:
    print("Can not import zero MQ")

class SUNEC13Commands:
    STATUS  = "STATUS"
    ON      = "ON"
    OFF     = "OFF"
    HON     = 'HON' # heater on
    HOFF    = 'HOFF' # heater on
    CON     = 'CON' # cooler on
    COFF    = 'COFF'
    TEMP0   = 'C1'
    TEMP1   = 'C2'
    TEMP2   = "C3"
    TEMP3   = "C4"
    SETTEMP = "SET"
    WAIT    = "WAIT" # wait time
    RATE    = "RATE" # temperature ramping time deg/minute
    STOP    = "STOP" # terminate run mode
    LTL     = "LTL"  # chamber lower temperature limit
    UTL     = "UTL"  # chamber upper temperature limit
    DEVL    = "DEVL" # chamber maximum deviation limit +/-
    PWMP    = "PWMP" # pwm period



    @staticmethod
    def querify(cmd):
        return cmd + "?"

    @staticmethod
    def settify(cmd):
        return cmd + "="


def setget(parameter, getter_only = False, doc=None):
    """
    Shortcut to construct property objects to wrap getters and setters
    for a number of settings

    Args:
        parameter: The parameter name to get/set. Get will be a query

    Returns:
        property object
    """
    if getter_only:
        return property(lambda self: self._get_parameter(parameter), doc=doc)
    else:
        return property(lambda self: self._get_parameter(parameter),\
                        lambda self, value: self._set_parameter(parameter,value), doc=doc)


class  SunChamber(object):

    axis_label = "Temperature "
    temperature_unit = "C"
    status_dict = [{ "Y" : "Power ON", "N" : "Power OFF"},\
                   { "Y" : "Last command error", "N" : "Last command ok"},\
                   { "Y" : "Time out LED ON", "N" : "Time out LED OFF"},\
                   { "Y" : "Waiting for timeout", "N" : "Not waiting for timeout"},\
                   { "Y" : "Heat output is enabled", "N" : "Heat output is disabled"},\
                   { "Y" : "Cool output is enabled", "N" : "Cool output is disabled"},\
                   { "Y" : "Valid set temperature", "N" : "Invalid set temperatur"},\
                   { "Y" : "Deviation limit exceeded", "N" : "Deviation limit ok"},\
                   { "Y" : "Currently ramping", "N" : "Not currently ramping"},\
                   { "Y" : "Chamber temp < lower limit", "N" : "Chamber temp > lower limit"},\
                   { "Y" : "Chamber temp > upper limit", "N" : "Chamber temp < upper limit"},\
                   { "Y" : "Waiting at a BKPNT", "N" : "Not waiting at a BKPNT"},\
                   { "Y" : "In LP run mode", "N" : "Not in LP run mode"},\
                   { "Y" : "In LP remote store mode", "N" : "Not in LP remote store mode"},\
                   { "Y" : "In local edit LP mode", "N" : "Not in local edit LP mode"},\
                   { "Y" : "Waiting to run LP at TOD", "N" : "Not waiting to run LP as TOD"},\
                   { "Y" : "GPIB bus timeout", "N" : "No GPIB bus timeout"},\
                   { "Y" : "In local keyboard lockout mode", "N" : "Not in local keyboard lockout mode"},\
                   { "0" : "System self test was ok", "1" : "Battery RAM error found (check default settings)",\
                     "2" : "EE RAM error found (check default settings)", "3" : "ROM error found (call factory)"}]
    

    #def __init__(self,gpib_adress=6, port=9999, publish=False):
    def __init__(self, controller, port=9999, publish=False):
        
        #@assert (isinstance(controller, NI_GPIB_USB) or isinstance(controller, PrologixUsbGPIBController)), "The used controller has to be either the NI usb one or the prologix usb"
        
        self.chamber = controller
        self.is_running = False
        self.publish = publish
        self.port = port
        self._socket = None

        # sometimes the chamber needs a bit till it is responding
        # get the status a few times
        self.get_temperature()
        self.get_status()
        self.get_status()

        self.last_status = ""
        status = self.get_status()
        self.print_status(status)
        if publish:
            self._setup_port()

    def _setup_port(self):
        """
        Setup the port for publishing

        Returns:
            None
        """
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        self._socket.connect("tcp://0.0.0.0:%s" % int(self.port))
        return

    def _set_parameter(self, parameter, value):
        command = f"{parameter}={value}\r\n"
        self.chamber.write(command)

    def _get_parameter(self, parameter):
        command = f"{parameter}?\r\n"
        resp = self.chamber.query(command)
        return resp
        
    # a bunch of setters/getters
    temperature_as_set = setget(SUNEC13Commands.SETTEMP)
    rate_as_set = setget(SUNEC13Commands.RATE)


    @property
    def ON(self):
        print ("Turning on chamber...")
        self.chamber.write(SUNEC13Commands.ON)

    @property
    def OFF(self):
        print ("Turning chamber off...")
        if self.is_running:
            print ("WARNING, will not turn off chamber whie it is operationg!!...")
            return
        self.chamber.write(SUNEC13Commands.OFF)

    def activate_heater(self):
        self.chamber.write(SUNEC13Commands.HON)
    
    def deactivate_heater(self):
        self.chamber.write(SUNEC13Commands.HOFF)

    def activate_cooler(self):
        self.chamber.write(SUNEC13Commands.CON)

    def deactivate_cooler(self):
        self.chamber.write(SUNEC13Commands.COFF)

    def get_status(self):
        self.last_status = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.STATUS))
        return self.last_status

    def _bit_io_channel_active(self, channel):
        """
        Read the in-built analog_io port, which controls a bunch of relays
        
        Args:
            channel (int) ; channel on the analog port, FIXME: which ic which is not yet clear
        """
        command = f"IN0:{channel},I0\r\n" # read the value into I0 variable
        self.chamber.write(command)
        command = "I0?"
        resp = self.chamber.query(command)
        print (resp)
        # FIXME: check which corresponds to actual on/off states
        if float(resp) == 1: # TTL low, input closed
            return False
        if float(resp) == 0: # TTL high, input open
            return True

    def _activate_bitio_channel(self, channel):
        """
        Write to the analog i/0 board
        """
        command = f"OUT0:{channel},1\r\n" # read the value into I0 variable
        self.chamber.write(command)       

    def _deactivate_bitio_channel(self, channel):
        """
        Write to the analog i/0 board
        """
        command = f"OUT0:{channel},0\r\n" # read the value into I0 variable
        self.chamber.write(command)

    def open_dry_nitrogen_valve(self):
        """
        The dry nitrogen valve needs to be opened during the warm up
        process to avoid humidity
        """
        self._activate_bitio_channel(4)

    def close_dry_nitrogen_valve(self):
        self._deactivate_bitio_channel(4)


    def cooldown(self, target_temperature=-45, rate=3):
        """
        A shortcut function to cool down the chamber to -45 deg 
        with about 3 deg per minute

        """
        if (target_temperature > 0):
            raise ValueError("Cooldown function is meant to be used with temperatures < 0, because of the dry nitrogen valve which will NOT be opened by this function! Interior of the chamber might get too humid...")
        self.rate_as_set = rate
        self.temperature_as_set = target_temperature
        current_temperature = self.get_temperature()
        start = time.monotonic()
        while current_temperature > (target_temperature + 2):
            now = time.monotonic() - start
            print ("Current temperature is {} C after {:4.2f} sec cooldown".format(current_temperature, now))
            time.sleep(5)

        return None

    @staticmethod
    def print_status(status): 
        print ("SUN EC13 chamber reporting status....")
        status = status.rstrip()
        for i,k in enumerate(status):
                print (SunChamber.status_dict[i][k])
        print ("----------------------------------")
    
    def show_status(self):
        status = self.get_status()
        self.print_status(status)

    def get_temperature(self, channel=0):
        """
        Channel 0,1
        """
        if channel == 0:
            temp = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.TEMP0))
        elif channel == 1:
            temp = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.TEMP1))
        elif channel == 2:
            temp = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.TEMP2))
        elif channel == 3:
            temp = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.TEMP3))
        else:
            raise ValueError("Channel has to be either 0,1,2 or 3!")
        #print ("Got channel temp of {}".format(temp))
        
        try:
            temp = float(temp)
        except ValueError:
            #print ("Problems digesting {}".format(temp))
            temp = np.nan

        if self.publish and (self._socket is None):
            self._setup_port()
        if self.publish:
            self._socket.send_string("{}::CH{} {}; {}".format("SUNEC13", channel, temp, self.temperature_unit))
  
        return temp 

    def measure_continously(self, npoints, interval):
        for n in range(npoints):
            temp1 = self.get_temperature(channel=0)
            temp2 = self.get_temperature(channel=1)
            if self.publish:
                self._socket.send_string("{}::CH{} {}; {}".format("SUNEC13", 0, temp1, self.temperature_unit))
                self._socket.send_string("{}::CH{} {}; {}".format("SUNEC13", 1, temp2, self.temperature_unit))
            time.sleep(interval)
            yield n*interval, (temp1,temp2)

    def monitor_temperatures(self, maxtime=np.inf):
        """
        Graphical representation of temperatures. If run in jupyter notebook
        the respective cell must include a %matplotlib notebook magic
        
        Keyword Args:
            maxtime (float): Maximum time the plot is active (in sec)
        """
        time_since_running = 0
        fig = p.figure(dpi=150)
        ax = fig.gca()
        ax.set_xlabel("time since start [s]")
        ax.set_ylabel("temperature $^{\circ}$C")
        p.ion()
        line_plots = [ax.plot(range(0), color=k, lw=3)[0] for k in ("r", "b", "g")]

        start_time = time.monotonic()

        fig.show()
        fig.canvas.draw()
        fig.tight_layout()
        while True:
            sec = time.monotonic() - start_time
            datamins, datamaxes = [],[]
            for ch, line_plot in enumerate(line_plots):
            
                temp = self.get_temperature(ch)
                secs, temps = line_plot.get_data()
                secs = np.append(secs,sec)
                temps = np.append(temps,temp)
                line_plot.set_ydata(temps)
                line_plot.set_xdata(secs)
                datamins.append(min(temps))
                datamaxes.append(max(temps))
                xmin = min(secs)
                xmax = max(secs)
                
            datamax = max(datamaxes)
            datamin = min(datamins)
            #print(secs)
            if len(secs) == 1:
                continue
        
            # avoid matplotlib warning
            if abs(datamin - datamax) < 1:
                datamin -= 1
                datamax += 1

            if abs(xmax - xmin) < 1:
                xmin -= 1
                xmax += 1

            # update the plot
            ax.set_xlim(xmin=xmin, xmax=xmax)
            ax.set_ylim(ymin=datamin, ymax=datamax)

            fig.tight_layout()
            fig.canvas.draw()
            time.sleep(5)
            time_since_running += 5

            if time_since_running > maxtime:
                return

        
