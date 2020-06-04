"""
A bunch of implementations to deal with temperature/humidity senors connected
to raspberry pis. This will typically run on a different machine than the rest 
of the setup, so the implementation is server-client. 
Most of the code is rather experiemental, and includes thermometers with multiple channels and also such which directly can draw a matplotlib plot in a gui.
"""

import time
import re
import hjson
import collections
import datetime
import zmq
import hepbasestack as hep

import matplotlib.dates as mdates

from .abstractbaseinstrument import AbstractBaseInstrument
from .. import controllers as contr
#from .. import controllers import GPIOController

try:
    from ..plugins.dht22 import adafruit_dht22_getter
except (ImportError, ModuleNotFoundError):
    adafruit_dht22_getter = lambda x : None
    print ("Can not import Adafruit_DHT")

#######################################################################

class RPIMultiChannelThermometer:
    """
    A thermometer build out of a raspberry pi + variaous sensors
    Base class. Use either server or client implementation
    """

    @staticmethod
    def _setup_serverside_socket(port):
        """
        Setup the port for publishing

        Returns:
            None
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://0.0.0.0:%s" % int(port))
        return socket
 



######################################################################

class RPIMultiChannelThermometerServer(RPIMultiChannelThermometer):


    def __init__(self,
                 loglevel=20,\
                 data_getters=(None,),\
                 data_getter_args=(None,),\
                 data_getter_kwargs=(None,),\
                 topics = (None,),\
                 publish_port=9876):
        """
        Constructor needs read and write access to
        the port

        Keyword Args:
            data_getters (tuple)       : a tuple of functions which are used 
                                         to obtain the raw data from the sensor.
            data_getter_args (tuple)   : Call data getters with this kwargs
            data_getter_kwargs (tuple) : Call data getters with this kwargs
            loglevel (int)             : 10: debug, 20: info, 30: warnlng....
            publish_port (int)         : Open a zmq socket for this port and publish
                                         the data on it for everybody to read. 
            topcis (tuple)             : A topic is a unique string which is published when 
                                         the data is published on the port to identify which    
                                         sensor send the data.
                                         Should be something descriptive, e.g. DRYBOX.
        """
        assert len(data_getters) == len(data_getter_args) == len(data_getter_kwargs) == len(topics), "Make sure each data getter has the appropriate args and kwargs and a topic"
        super(RPIMultiChannelThermometerServer, self).__init__()

        self.logger = log = hep.logger.get_logger(loglevel)
        self.data_getters = data_getters
        self.data_getter_args = data_getter_args
        self.data_getter_kwargs = data_getter_kwargs
        self.topics = topics
        self.publish = False
        if publish_port is not None:
            self._socket = self._setup_serverside_socket(publish_port)
            self.publish = True
        self.logger.debug("Instrument initialized")

    def __del__(self):
        if self.publish:
            self._socket.close()

    def pull_data(self):
        """
        Use the data getters to pull data from the sensors. If the publish setting is given,
        data will be published at the given port.
        """
        for i,getter in enumerate(self.data_getters):
            args = [] if (self.data_getter_args[i] is None) else self.data_getter_args[i]
            kwargs = {} if (self.data_getter_kwargs[i] is None) else self.data_getter_kwargs[i]
            data = getter(*args, **kwargs)
            if self.publish:
                self._socket.send((self.topics[i] + "\t" + data).encode())
            yield data

    def measure_continuously(self, measurement_time=10, interval=5):
        """
        Do a continuous measurment

        Keyword Args:
            measurment_time (int) : measurment duration [seconds], can be np.inf
            interval (int)        : measurment interval [seconds]

        """
        last_step = time.monotonic()
        passed_time = 0
        data = dict([(t,"") for t in self.topics])
        while passed_time < measurement_time:
            for i,result in enumerate(self.pull_data()):
                data[self.topics[i]] = result
            time.sleep(interval)
            dt = time.monotonic() - last_step
            last_step = time.monotonic()
            passed_time += dt
            yield data
            #if payload is None:
            #    self.logger.warning("Can not get data {}".format(payload))
            #    continue
            #if self.publish:
            #    self._socket.send((self.TOPIC + "\t" + payload).encode())
            #self.logger.debug("Got data {}".format(payload))
            #yield payload

#######################################################################

class RPIMultiChannelThermometerClient(RPIMultiChannelThermometer):
    pass


#######################################################################

class RaspberryPiGPIODHT22Thermometer(AbstractBaseInstrument):
    """
    A custom build with 4 DHT22 temperature sensor
    and a raspberry pi
    """
   
    # define a pattern which defines this datastram
    PATTERN =re.compile("(?P<timestamp>[A-Za-z0-9\s:]*)\s(?P<s1temp>[0-9\.-]*)\s(?P<s1humi>[0-9\.]*)\s(?P<s2temp>[0-9\.-]*)\s(?P<s2humi>[0-9\.]*)\s(?P<s3temp>[0-9\.-]*)\s(?P<s3humi>[0-9\.]*)\s(?P<s4temp>[0-9\.-]*)\s(?P<s4humi>[0-9\.]*)")   
    TYPES  = { "timestamp"  : lambda x: datetime.datetime.strptime(x,"%a %b %d %H:%M:%S %Y"),
                "s1temp"    : lambda x: float(x),
                "s1humi"    : lambda x: float(x),
                "s2temp"    : lambda x: float(x),
                "s2humi"    : lambda x: float(x),
                "s3temp"    : lambda x: float(x),
                "s3humi"    : lambda x: float(x),
                "s4temp"    : lambda x: float(x),
                "s4humi"    : lambda x: float(x)}

    # define the order of sorting
    PAYLOAD  = collections.OrderedDict([("timestamp" , ""),
                                        ( "s1temp"   , ""),
                                        ( "s1humi"   , ""),
                                        ( "s2temp"   , ""),
                                        ( "s2humi"   , ""),
                                        ( "s3temp"   , ""),
                                        ( "s3humi"   , ""),
                                        ( "s4temp"   , ""),
                                        ( "s4humi"   , "")])

    TOPIC  = "FREEZER"            
    METADATA = { "name"           : "RaspberryPiDHT22-4Channel",\
                 "twinax"         : True,\
                 "units"          : ["C", "\%"],\
                 "axis_labels"    : ["Temp C", "Hum \%"],\
                 "xdata"          : "timestamp",\
                 "plot_type"      : "date",\
                 "channels"       : ["s1temp", "s2temp", "s3temp", "s4temp"],\
                 "twinaxchannels" : ["s1humi", "s2humi", "s3humi", "s4humi"],\
                 "xmaj_formatter" : mdates.DateFormatter("%m/%d/%Y"),\
                 "xmaj_locator"   : mdates.DayLocator(),\
                 "xmin_formatter" : mdates.DateFormatter("%H:%M"),\
                 "xmin_locator"   : mdates.HourLocator(),
               }
    REVERSE_STRING_TEMPLATE = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
    

    def __init__(self,
                 controller=contr.GPIOController(data_getter=adafruit_dht22_getter, data_getter_kwargs={"pins" : [4,14,17,24]}),\
                 loglevel=20,\
                 publish=False,
                 publish_port=9876):
        super(RaspberryPiGPIODHT22Thermometer, self).__init__(controller=controller, loglevel=loglevel,
                                                              publish=publish, publish_port=publish_port)
        if publish:
            self._setup_port()    

    def decode_payload(self, data):
        """
        Go from string to dict
        """
        if data is None:
            self.logger.warning("Empty data")
            return None
        payload = self.PATTERN.search(data)
        if payload is None:
            self.logger.warning("Can not decode data {}".format(data))
            return None
        payload = payload.groupdict()
        sorted_payload = self.PAYLOAD
        for k in sorted_payload:
            sorted_payload[k] = payload[k]
        return sorted_payload

    @staticmethod
    def encode_payload(payload, reverse_string_template=None):
        """
        Go from dict back to string
        """
        if payload is None:
            return None
        payload = hjson.loads(payload)
        reverse_string = reverse_string_template.format(*payload.values())
        return reverse_string

    def read(self):
        payload = self.decode_payload(self.controller.read())
        if payload is None:
            return None
        return hjson.dumps(payload)




    def measure(self):
        payload = self.read()
        if payload is None:
            self.logger.warning("Can not get data {}".format(payload))
            return None
        if self.publish:
            self._socket.send((self.TOPIC + "\t" + payload).encode())
        return payload


    def measure_continuously(self, measurement_time=10, interval=5):
        """
        Do a continuous measurment

        Keyword Args:
            measurment_time (int) : measurment duration [seconds], can be np.inf
            interval (int)        : measurment interval [seconds]

        """
        start = time.monotonic()
        delta_t = 0
        while delta_t < measurement_time:
            payload = self.read()
            time.sleep(interval)
            delta_t += (time.monotonic()  - start)
            if payload is None:
                self.logger.warning("Can not get data {}".format(payload))
                continue
            if self.publish:
                self._socket.send((self.TOPIC + "\t" + payload).encode())
            self.logger.debug("Got data {}".format(payload))
            yield payload
       
class RaspberryPiGPIODHT22ThermometerSingleChannel(RaspberryPiGPIODHT22Thermometer):
    """
    A custom build with a single DHT22 temperature sensor
    and a raspberry pi
    """
   
    # define a pattern which defines this datastram
    PATTERN =re.compile("(?P<timestamp>[A-Za-z0-9\s:]*)\s(?P<s1temp>[0-9\.-]*)\s(?P<s1humi>[0-9\.]*)")   
    TYPES  = { "timestamp"  : lambda x: datetime.datetime.strptime(x,"%a %b %d %H:%M:%S %Y"),
                "s1temp"    : lambda x: float(x),
                "s1humi"    : lambda x: float(x)}

    # define the order of sorting
    PAYLOAD  = collections.OrderedDict([("timestamp" , ""),
                                        ( "s1temp"   , ""),
                                        ( "s1humi"   , "")])

    TOPIC  = "SUNEC13"
    TOPIC  = "SUNEC13"            
    METADATA = { "name"           : "RaspberryPiDHT22-1Channel",\
                 "twinax"         : True,\
                 "units"          : ["C", "\%"],\
                 "axis_labels"    : ["Temp C", "Hum \%"],\
                 "xdata"          : "timestamp",\
                 "plot_type"      : "date",\
                 "channels"       : ["s1temp"],\
                 "twinaxchannels" : ["s1humi"],\
                 "xmaj_formatter" : mdates.DateFormatter("%m/%d/%Y"),\
                 "xmaj_locator"   : mdates.DayLocator(),\
                 "xmin_formatter" : mdates.DateFormatter("%H:%M"),\
                 "xmin_locator"   : mdates.HourLocator(),
               }
    REVERSE_STRING_TEMPLATE = "{}\t{}\t{}"

    def __init__(self,
                 controller=contr.GPIOController(data_getter=adafruit_dht22_getter, data_getter_kwargs={"pins" : [4]}),\
                 loglevel=20,\
                 publish=False,
                 publish_port=9876):
        super(RaspberryPiGPIODHT22Thermometer, self).__init__(controller=controller, loglevel=loglevel,
                                                              publish=publish, publish_port=publish_port)
        if publish:
            self._setup_port()
#class RaspberryPiGPIODHT22ThermometerProxy(RaspberryPiGPIODHT22Thermometer):
#    """
#    Connect to a remotly running RaspberryPiGPIODHT22Thermometer instance
#    """
#    def __init__(self,ip):
#        # Socket to talk to server
#        self.context = zmq.Context()
#        self.socket = self.context.socket(zmq.SUB)
#        
#        socket.connect ("tcp://{}:{}".format(ip,port))
#        
#        topicfilter = RaspberryPiGPIODHT22Thermometer.TOPIC.encode()
#        socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
#
#    def read(self):
#        payload = self.socket.recv().decode()
#        print (payload)
#
#    def measure_contiuously(self.measure_time=10, interval=5)
       
        
