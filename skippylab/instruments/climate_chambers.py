"""
Control SUN EC13 temperature chamber. This device can maintain a certain
temperature and drive temperature profiles. 
"""

import numpy as np
import time 
import skippylab
import pylab as p
import hepbasestack as hbs

# visual adaptions for the jupyter notebook
hbs.visual.set_style_default()

from .. controllers import PrologixUsbGPIBController

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


class SunChamber(object):
    """
    The Sun EC13 temperature chamber allows for a temperature controlled environment.
    """

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
    

    def __init__(self, controller, port=9999,
                 publish=False, logger=None,\
                 loglevel=skippylab.LOGLEVEL):
        """
        Open the connection to a SUN EC13 climate chamber via a controller. 
        
        Args:
            controller (skippylab.controllers.AbstractBaseController)  :  A gpib or whichever instance 
                                                                          used to connect to the cmaber
        Keyword Args:
            port (int)              : A network port which is used to publish chamber 
                                      sensor data if publish = True
            publish (bool)          : Publish the temperature data of the internal sensors
                                      on the network
            logger (logging.Logger) : Use the logger instance to publish status messages
                                      if None, a new logger is created with level loglevl
            loglevel (int)          : The loglevel in case logger is not None
        """
        
        self.chamber = controller
        self.publish = publish
        self.port = port
        self._socket = None
        if logger is None:
            self.logger = hbs.logger.get_logger(loglevel)
        else:
            self.logger = logger
        self.logger.debug("Initializing chamber")
        # sometimes the chamber needs a bit till it is responding
        time.sleep(1)

        self.last_status = ""
        status = self.get_status()
        self.last_status = status
        #self.print_status(status)
        if publish:
            self._setup_port()

    def __del__(self):
        del self.chamber


    def _setup_port(self):
        """
        Setup the port for publishing

        Returns:
            None
        """
        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        self._socket.connect("tcp://0.0.0.0:%s" % int(self.port))
        return Nonen

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
    upper_temperature_limit_as_set = setget(SUNEC13Commands.UTL)
    lower_temperature_limit_as_set = setget(SUNEC13Commands.LTL)

    @property
    def is_on(self):
        status = self.get_status(nofail=True)
        return status[0] == "Y" 

    @property
    def ON(self):
        self.logger.info("Turning on chamber...")
        self.chamber.write(SUNEC13Commands.ON)

    @property
    def OFF(self):
        self.logger.info("Turning chamber off...")
        self.chamber.write(SUNEC13Commands.OFF)

    def activate_heater(self):
        """
        Activate the internal heating system
        """
        self.chamber.write(SUNEC13Commands.HON)
    
    def deactivate_heater(self):
        """
        Switch off the internal heating system
        """
        self.chamber.write(SUNEC13Commands.HOFF)

    def activate_cooler(self):
        """
        This allows the chamber to draw liquid N2 to cool itself down.
        """

        self.chamber.write(SUNEC13Commands.CON)

    def deactivate_cooler(self):
        """
        Closes the valve for liquid N2, prevents the chamber from drawing 
        anymore LN2
        """

        self.chamber.write(SUNEC13Commands.COFF)

    def get_status(self, nofail=False):
        """
        Make the chamber report on its internal status. Issues status command
        to the chamber and parses return.

        Keyword Args:
            nofail (bool) : if True, force an answer, that is the chamber will be asked
                            for status until it returns it.
        """
        self.logger.debug("Getting status...")
        status = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.STATUS))
        if nofail:
            while not status:
                status = self.chamber.query(SUNEC13Commands.querify(SUNEC13Commands.STATUS))
        self.last_status = status
        self.logger.debug('... done.')
        return status


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
        self.logger.debug(f'Recieved response {resp}')
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
        The chamber has an additonal valve with a manual flow meter to allow
        the purge with gaseous nitrogen. This opens this valve.
        """
        self._activate_bitio_channel(4)

    def close_dry_nitrogen_valve(self):
        """
        Close the valve for gaseous nitrogen, so that the chamber can 
        not draw any dry nitrogen.
        """
        self._deactivate_bitio_channel(4)


    def cooldown(self, target_temperature=-37, rate=3, 
                 channel_for_monitoring = 0):
        """
        A shortcut function to cool down the chamber to -45 deg 
        with about 3 deg per minute

        Keyword Args:
            channel_for_monitoring (int): use this temperature 
                                          channel for monitoring

        """
        if (target_temperature > 0):
            raise ValueError("Cooldown function is meant to be used with temperatures < 0, because of the dry nitrogen valve which will NOT be opened by this function! Interior of the chamber might get too humid...")
        self.rate_as_set = rate
        # set the set temperature a little bit lower
        # than the target temperature to make sure it gets reached
        self.temperature_as_set = target_temperature - 3
        current_temperature = self.get_temperature(channel_for_monitoring)
        start = time.monotonic()
        while current_temperature > (target_temperature):
            now = time.monotonic() - start
            self.logger.debug("Current temperature is {} C after {:4.2f} sec cooldown".format(current_temperature, now))
            time.sleep(5)
            current_temperature - self.get_temperature(channel_for_monitoring)
        self.temperature_as_set = target_temperature
        return None

    @staticmethod
    def print_status(status): 
        print("SUN EC13 chamber reporting status....")
        status = status.rstrip()
        for i,k in enumerate(status):
                print(f"{SunChamber.status_dict[i][k]}")
        print("------------DONE----------------------")
    
    def show_status(self):
        status = self.get_status()
        self.print_status(status)

    def get_last_status(self):
        status_string = ""
        status = self.last_status.rstrip()
        for i,k in enumerate(status):
            status_string += self.status_dict[i][k] + "\n"
        return status_string

    def get_temperature(self, channel=0, timeout = 8):
        """
        Channel 0,1
        """
        def query_temps(channel):

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
                temp = np.nan
            return temp 
        
        temp = query_temps(channel)
        start = time.monotonic()
        passed_time = 0
        while np.isnan(temp):
            temp = query_temps(channel)
            time.sleep(0.5)
            end  = time.monotonic()
            passed_time = end - start
            #start = time.monotonic() - start
            if passed_time > timeout:
                raise TimeoutError(f"Can not get a value for temperature within {timeout}")
                break
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

    def monitor_temperatures(self,
                             maxtime=np.inf,
                             target_temp=None,
                             activate=False, 
                             feedback_ch=0):
        """
        Graphical representation of temperatures. If run in jupyter notebook
        the respective cell must include a %matplotlib notebook magic
        This function is blocking until either maxtime or target_temp is reached.
        
        Keyword Args:
            maxtime (float)    : Maximum time the plot is active (in sec)
            target_temp (float): Return if target_temp is reached
            activate (bool)    : if True, try to reach the target temperature, not only do monitoring 
            feedback_temp (int): Use this channel for feedback for the temperature loop as monitoring 
        """
        time_since_running = 0
        fig = p.figure(dpi=150)
        ax = fig.gca()
        ax.set_xlabel("time since start [s]")
        ax.set_ylabel("temperature $^{\circ}$C")
        p.ion()
        line_plots = [ax.plot(range(0), color=k, lw=3)[0] for k in ("r", "b")]

        start_time = time.monotonic()

        fig.show()
        fig.canvas.draw()
        fig.tight_layout()

        feedback_temp = np.nan
        if activate:
            self.temperature_as_set = target_temp
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
                if ch == feedback_ch:
                    feedback_temp = temp
            
            if abs(abs(target_temp) - abs(feedback_temp)) < 1:
                self.logger.info(f"Reached target temperature of {target_temp}, feedback temp is reading {feedback_temp}") 
                return

            datamax = max(datamaxes)
            datamin = min(datamins)
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
            ax.set_xlim(left=xmin, right=xmax)
            ax.set_ylim(bottom=datamin, top=datamax)

            fig.tight_layout()
            fig.canvas.draw()
            time.sleep(5)
            time_since_running += 5

            if time_since_running > maxtime:
                return


