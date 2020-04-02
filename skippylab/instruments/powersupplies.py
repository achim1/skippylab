"""
Control of different benchtop orNIM power supplies

"""
import time


from . import oscilloscopes as osci
from ..scpi import commands as cmd

from hepbasestack.logger import get_logger
from hepbasestack.logger import Logger as _RootLogger
import hepbasestack as hbs

import pylab as p

# set styles for notebook
hbs.visual.set_style_default()

bar_available = False

try:
    import tqdm
    bar_available = True
except ImportError:
    _RootLogger.warn("Module tqdm not available, disabeling progress bars...")

try:
    from plx_gpib_ethernet import PrologixGPIBEthernet
except ImportError as e:
    _RootLogger.warn('No plx_gpib_ethernet module installed')

setget = osci.setget
KCmd = cmd.KeysightE3631APowerSupplyCommands

q = cmd.query

class KeysightE3631APowerSupply(object):
    """
    A low voltage power supply with two channels, +6V and +- 25V manufactured
    by Keysight. The power supply does not have an ethernet port, so the
    connection is done via GPIB and a prologix GPIB Ethernet connector
    """
    output = setget(KCmd.OUTPUT)

    def __init__(self, ip="10.25.124.252", gpib_address=5, loglevel=20):
        """
        Connect to the power supply via Prologix GPIB connector

        Keyword Args:
            ip (str): IP adress of the Prologix GPIB connector
            gpib_address (int): The GPIB address of the power supply
                                connected to the Prologix connector
        """
        gpib = PrologixGPIBEthernet(ip)
        gpib.connect()
        gpib.select(gpib_address)
        self.logger = get_logger(loglevel)
        self.instrument = gpib
        self.P6 = KCmd.P6
        self.P25 = KCmd.P25
        self.N25 = KCmd.N25

    def __del__(self):
        self.instrument.close()

    def _query(self, command):
        """
        Send a command

        Args:
            command (str):

        Returns:
            str
        """
        self.logger.debug("Querying {}".format(command))
        return self.instrument.query(command)

    def _set(self, command):
        """
        Send a command bur return no response

        Args:
            command (str): command to be send to the scope

        Returns:
            None
        """

        #FIXME: Make AbstractInstrment class and inherit
        self.logger.debug("Sending {}".format(command))
        self.instrument.write(command)

    def ping(self):
        """
        Check the connection

        Returns:
            str
        """

        return self.instrument.query(cmd.WHOAMI)

    @property
    def error_state(self):
        """
        Read out the error register of the power supply

        Returns:
            str
        """

        error =  self.instrument.query(KCmd.ERROR_STATEQ)
        error = error.split(",")
        err_no = float(error[0])
        return err_no, "".join(error[1:])

    def select_channel(self, channel):
        """
        Select either the +6, +25 or -25V channel

        Args:
            channel (str or int):

        Returns:
            None
        """
        channel_dict = {1: self.P6, 2: self.P25, 3: self.N25}
        if isinstance(channel, int):
            assert channel in [1, 2, 3], "Channel has to be either 0:+6V, 1:+25V or 2: -25V"
            channel = channel_dict[channel]
        elif isinstance(channel, str):
            assert channel in (KCmd.P6, KCmd.P25, KCmd.N25),\
                "Channel has to be in {}".format(KCmd.P6, KCmd.P25, KCmd.N25)
        else:
            raise ValueError("Channel has to be either str or int")

        self._set(KCmd.CHANNEL + " {}".format(channel))
        return channel

    def set_voltage(self, channel, voltage):
        """
        Set the supplied voltage of a channel to the desired value

        Args:
            channel (str or int):
            voltage (float):

        Returns:
            None
        """
        channel = self.select_channel(channel)
        if (channel == self.P6) and (voltage > 6.18):
            raise ValueError("6V Channel does not support {}".format(voltage))
        self._set(KCmd.VOLT + " {}".format(voltage))

    def off(self):
        """
        Cut the power on all channels

        Returns:
            None
        """
        self.logger.info("Disabling power...")
        self.output = KCmd.OFF

    def on(self):
        """
        Enable power on all channels

        Returns:
            None
        """
        self.logger.warning("Enabling power!!")

        # give the user time to hit Ctrl-C to abort
        safety_wait = 5
        if bar_available:
            for i in tqdm.tqdm(range(safety_wait)):
                time.sleep(1)
        else:
            self.logger.warning("Powering on in {}".format(safety_wait - i))
        self.output = KCmd.ON

    def measure_current(self, channel):
        """
        Measure current on givven channel

        Args:
            channel (str):

        Returns:
            float
        """
        command = q(KCmd.MEASURE + ":" + KCmd.CURRENT + ":" + KCmd.DC)
        command += " {}".format(channel)
        return float(self._query(command))


######################################################################################################


import serial
import time
import tqdm
import re
import numpy as np


class ResponseException(Exception):
    pass

class CommandException(Exception):
    pass

class ChannelException(Exception):
    pass

class ParameterException(Exception):
    pass

class ValueException(Exception):
    pass

class LocalModeError(Exception):
    pass

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


class CAENN1471ChannelCommands(object):
    RAMP = "RAMP"
    KILL = "KILL"
    LOW  = "LOW"
    HIGH = "HIGH"

class CAENN1471BoardCommands(object):
    CLOSED = "CLOSED"
    OPEN = "OPEN"

class Channel(object):
    """
    Namespace to access the individual channels of
    the CAENN1471 HV power supply
    """
    def __init__(self, channel, board, loglevel=30, time_delay=0.1):
        """
        Keyword Args:
            loglevel (int)     : 10 dbg, 20 info, 30 warn
            time_delay (float) : set a time delay to await the 
                                 response. If this is too short
                                 the responses might be garbage 
        """
        self.channel = channel
        self.board = board
        self.logger = get_logger(loglevel)
        self.logger.debug(f"Initialize hannel {channel} ...")
        self.time_delay = time_delay
        self.logger.debug("done!")
    # per channel commands, set parameters
    def _set_parameter(self, parameter, value):
        command = "$BD:{:02d},CMD:SET,CH:{},PAR:{},VAL:{}".format(self.board.board,self.channel,parameter, str(value))
        self.board._send(command)
        response = self.board._listen()
        self.logger.debug("Set parameter for channel {} succesful {}".format(self.channel, response))    

    # get channel values
    def _get_parameter(self, parameter):
        command = "$BD:{:02d},CMD:MON,CH:{},PAR:{}".format(self.board.board,self.channel,parameter)
        self.board._send(command)
        response = self.board._listen()
        self.logger.debug("Retrieved parameter {} for channel {}".format(response, self.channel, response))    
        return response    
   
    def activate(self, nonblocking=False):
        """
        Set channel to on 
    
        Keyword Args:
            nonblocking (bool) : if True, return immediatly without any return value.

        """
        self.logger.info("Activating channel {}".format(self.channel))
        command = "$BD:{:02d},CMD:SET,CH:{},PAR:ON".format(self.board.board, self.channel)
        self.board._send(command)
        self.board._listen()
        time_since_start = time.monotonic()
        # wait till it is ramped-up
        rate = self.ramp_up
        delta_v = np.array(self.voltage_as_set) - np.array(self.voltage_as_is)
        wait = int(delta_v/rate)
        if nonblocking:
            return None
        voltages, currents, times = [], [], []
        loader_string = "Ramping up HV..."
        if hbs.isnotebook():
            bar = tqdm.tqdm_notebook(total=wait, desc=loader_string, leave=True)
        else:
            bar = tqdm.tqdm(total=wait, desc=loader_string, leave=True)
        for k in range(wait):
            voltages.append(self.voltage_as_is)
            currents.append(self.current_as_is)
            times.append(time.monotonic() - time_since_start)
            time.sleep(1)
            bar.update()
        return times, voltages, currents

    def take_iv_curve(self, voltages, time_interval=1):
        """
        Take iv curve. Get voltages and currents for the given 
        number of voltages.

        Args:
            voltages (iterable): array of voltages

        Keyword Args:
            time_interval (float): time [in sec] between measurments

        """
        command = "$BD:{:02d},CMD:SET,CH:{},PAR:ON".format(self.board.board, self.channel)


        measured_voltages = []
        measured_currents = []
        for volt in tqdm.tqdm(voltages):
            try:
                self.voltage_as_set = volt
                self.board._send(command)
                measured_voltages.append(self.voltage_as_is)
                measured_currents.append(self.current_as_is)
                time.sleep(time_interval)
            except Exception as e:
                print (e)
                time.sleep(2*time_interval)

        measured_voltages = np.array(measured_voltages)
        measured_currents = np.array(measured_currents)
        return measured_voltages, measured_currents

    def deactivate(self, nonblocking=False):
        """
        Set channel to off 

        Keyword Args:
            nonblocking (bool) : if True, return immediatly without return value
        """
        command = "$BD:{:02d},CMD:SET,CH:{},PAR:OFF".format(self.board.board, self.channel)
        self.board._send(command)
        self.board._listen()
        rate = self.ramp_down
        delta_v =  np.array(self.voltage_as_is)
        wait = int(delta_v/rate)
        if nonblocking:
            return None
        loader_string = "Ramping down channel..."
        if hbs.isnotebook():
            bar = tqdm.tqdm_notebook(total=wait, desc=loader_string, leave=True)
        else:
            bar = tqdm.tqdm(total=wait, desc=loader_string, leave=True)
        for __ in range(wait):
            time.sleep(1)
            bar.update()

    @property
    def status(self):
        command = "$BD:{:02d},CMD:MON,CH:{},PAR:STAT".format(self.board.board, self.channel)
        self.board._send(command)
        response = self.board._listen()
        return response
        

    def monitor(self, maxtime=np.inf):
        """
        Monitor current and voltages in the ipython notebook 

        Keyword Args:
            maxtime (float): End function after maxtime in secs

        """
        fig = p.figure(dpi=150)
        ax = fig.gca()
        ax.set_xlabel("time since start [s]")
        ax.set_ylabel("voltage [V]")
        
        # current axis
        ax_current = ax.twinx()
        ax_current.set_ylabel("current [$\mu$A]")
        p.ion()
        line_plots = [ax.plot([0],[0], color='r', lw=3)[0],
                      ax_current.plot([0],[0], color='b', lw=3)[0]]
    
        start_time = time.monotonic()
        ax.spines["left"].set_visible(False)
        ax_current.spines["left"].set_visible(False)
        fig.show()
        fig.canvas.draw()
        fig.tight_layout() 
        while True:
            sec = time.monotonic() - start_time
            datamins, datamaxes = [],[]
            for ch, line_plot in enumerate(line_plots):
            
                value = self.voltage_as_is
                if (ch == 1):
                    # I am deeply sorry for this...
                    value = self.current_as_is

                secs, values = line_plot.get_data()
                secs = np.append(secs,sec)
                volts = np.append(values,value)
                line_plot.set_ydata(values)
                line_plot.set_xdata(secs)
                datamins.append(min(values))
                datamaxes.append(max(values))
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
    
            fig.canvas.draw()
            fig.tight_layout()
            time.sleep(1)
            if secs[-1] > maxtime:
                return


    # setters/getters
    voltage_as_set = setget("VSET", doc = "The desired voltage")
    current_as_set = setget("ISET", doc = "The desired current")
    max_set_voltage = setget("MAXV")
    ramp_up = setget("RUP")
    ramp_down = setget("RDW")
    trip_time = setget("TRIP")
    # can be either KILL or RAMP
    power_down_ramp_or_kill = setget("PDWN")
    # can be either LOW or HIGH
    imon_range_low_or_high = setget("IMRANGE")

    # getters only
    voltage_as_is = setget("VMON", getter_only = True, doc = "Get current [true] voltage")
    max_set_voltage = setget("VMAX", getter_only = True, doc = "Get max vset value")
    min_set_voltage = setget("VMIN", getter_only = True, doc = "Get min vset value")
    voltage_n_decimal_digits = setget("VDEC", getter_only = True)

    current_as_is = setget("IMON", getter_only = True, doc = "Get curretn [true] current")
    max_set_current = setget("IMAX", getter_only = True)
    min_set_current = setget("IMIN", getter_only = True)
    current_n_decimal_digits = setget("ISDEC", getter_only=True)

    polarity = setget("POL", getter_only=True, doc="channel polarity, readout only") 


class CAENN1471HV(object):
    #command_string = 
    ERROR_STRING = re.compile('#BD:[0-9]{2},(?P<errcode>[A-Z]{2,3}):ERR')
    SUCCESS_STRING = re.compile('#BD:[0-9]{2},CMD:OK(,VAL:)*(?P<values>[0-9A-Za-z;.]*)')

    def __init__(self, port='/dev/caen1471',
                 board=0,
                 time_delay=1,
                 loglevel=30):
        """
        Set up a connection to a CAEN1471HV module via usb/serial connection.

        Keyword Args:
            port (str):  unix device file, e.g. /dev/ttyUSB0 or /dev/caen1471
            board (int): board number
            loglevel (int) : 10 dbg, 20 info, 30 warn
        """
        self.logger = get_logger(loglevel)
        self.logger.info('Opening connection to {}'.format(port))
        self.connection = CAENN1471HV.open_connection(port)
        self.logger.info('Connection established!')
        self.board = board
        self.last_command = None
        # give the hv module some time to respond
        # wait self.time_delay after each write command 
        # on the interface
        self.time_delay = time_delay
        self.channels = dict()
        # the channel number 0-3 are the 4 channels
        # channel 4 is all channels together
        try:
            nchannels = int(self.nchannels[0]) + 1
        except Exception as e:
            time.sleep(0.5)
            nchannels = int(self.nchannels[0]) + 1
        
        for i in range(int(self.nchannels[0]) + 1):
            thischan = Channel(i, board=self, loglevel=loglevel)
            if i < 4:
                setattr(self, "channel{}".format(i),thischan)
            if i == 4:
                setattr(self, "all_channels", thischan)
            self.channels[i] = thischan

    def __del__(self):
        self.connection.close()

    def __repr__(self):
        return "<CAENN1471HV: {} board no: {} serial number {}>".format(self.boardname[0], self.board, self.serial_number)

    @staticmethod
    def open_connection(port):
        conn = serial.Serial(port=port, xonxoff=True)   
        loader_string = "Establishing connection..."
        if hbs.isnotebook():
            bar = tqdm.tqdm_notebook(total=10, desc=loader_string, leave=True)
        else:
            bar = tqdm.tqdm(total=10, desc=loader_string, leave=True)

        for __ in range(10):
            time.sleep(0.05)
            bar.update()
        return conn

    # low level read/write
    def _send(self,command):
        if not command.endswith("\r\n"):
            command += "\r\n"
        command = command.encode()
        self.last_command = command
        self.logger.debug("Sending command {}".format(command))
        self.connection.write(command)
        time.sleep(self.time_delay)

    def check_response(self,response):
        self.logger.debug(f"Checking response {response}")
        err = CAENN1471HV.ERROR_STRING.match(response)
        if err is not None:
            errcode = err.groupdict()["errcode"]
            if errcode == "CMD":
                raise CommandException("Wrong command format or command not recongized {}".format(self.last_command))
            elif errcode == "CH":
                raise ChannelException("Channel field not present or wrong channel value {}".format(self.last_command))
            elif errcode == "PAR":
                raise ParameterException("Field parameter not present or parameter not recognized {}".format(self.last_command))
            elif errcode == "VAL":
                raise ValueException("Wrong set value (<Min or >Max) {}".format(self.last_command))
            elif errcode == "LOC":
                raise LocalModeError("Command SET with module in LOCK mode {}".format(self.last_command))
            else:
                raise ResponseException("Some unknown problem occured with command {} - got response {}".format(self.last_command, response))

    def _listen(self):
        response = self.connection.read_all().decode()
        if not response:
            self.logger.debug("did not recieve response")
            return
        # FIMXE:
        self.check_response(response)
        success = CAENN1471HV.SUCCESS_STRING.match(response)
        if success is not None:
            self.logger.debug("Got repsonse {}".format(response))
            if 'values' in success.groupdict():
                self.logger.debug("Got values {}".format(success.groupdict()['values']))
                values = success.groupdict()['values']
                if ';' in values:
                    values = [k for k in values.split(';')]
                elif values:
                    values = [values]
                else:
                    return None
                try:
                    values = [float(k) for k in values]
                except ValueError: #it is a string then
                    pass
                return values
            return None
        else:
            raise ResponseException("Something went wrong, got garbage response {}".format(response)) 

    def _get_parameter(self,parameter):
        command = "$BD:{:02d},CMD:MON,PAR:{}".format(self.board, parameter)
        self._send(command)
        response = self._listen()
        return response

    def _set_parameter(self,parameter,value):
        command = "$BD:{:02d},CMD:MON,PAR:{},VAL:{}".format(self.board, parameter, value)
        self._send(command)
        response = self._listen()
        return response

    def clear_alarm_signal(self):
        command = "$BD:{:02d},CMD:SET,PAR:BDCLR".format(self.board)
        self._send(command)
        response = self._listen()

    interlock_mode = setget("BDILKM", doc="Either OPEN or CLOSED")
    boardname = setget("BDNAME", getter_only=True, doc="Name of the board")
    nchannels = setget("BDNCH", getter_only=True, doc="Number of provided channels")
    firmware_release = setget("BDFREL", getter_only=True, doc="Firmware release version number") 
    serial_number = setget("BDSNUM", getter_only=True, doc="Serial number of the board")
    interlock_status = setget("BDILK", getter_only=True, doc="Interlock status, either YES or NO") 
    control_mode = setget("BDCTR", getter_only=True, doc="Control mode, either REMOTE or LOCAL")
    local_bus_termination_status  = setget("BDTERM", getter_only=True, doc="Local bus termination status, either ON or OFF")
    board_alarm_bitmask = setget("BDALARM", getter_only=True, doc="Board alarm bitmask, needs to be decoded") 
      



