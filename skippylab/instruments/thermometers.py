import time
import re
import hjson
import collections
import datetime

import matplotlib.dates as mdates

from .abstractbaseinstrument import AbstractBaseInstrument
from ..controllers import GPIOController
try:
    from ..plugins.dht22 import adafruit_dht22_getter
except (ImportError, ModuleNotFoundError):
    adafruit_dht22_getter = lambda x : None
    print ("Can not import Adafruit_DHT")

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
    

    def __init__(self,
                 controller=GPIOController(data_getter=adafruit_dht22_getter, data_getter_kwargs={"pins" : [4,14,17,24]}),\
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
    def encode_payload(payload):
        """
        Go from dict back to string
        """
        if payload is None:
            return None
        payload = hjson.loads(payload)
        reverse_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(*payload.values())
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
            time.sleep(interval)
            delta_t += (time.monotonic()  - start)
            payload = self.measure()
            if payload is None:
                self.logger.warning("Can not get data {}".format(payload))
                continue
            self.logger.debug("Got data {}".format(payload))
            yield payload
        
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
       
        
