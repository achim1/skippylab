"""
The DHT22 is a very simple, but popular temperature/humidity sensor.
It can be polled every few seconds and has an accuracy of +- 1deg C
and +- 5% relative humidity.
"""
import datetime
import time
import hepbasestack.logger as log
import skippylab
Logger = log.get_logger(skippylab.logleve)

HAS_DHT = False
try:
    import Adafruit_DHT
    HAS_DHT = True
except ModuleNotFoundError:
    Logger.warning("Can not use adafruit dht sensors on this machine. Try installing requirements with pip3 install Adafruit_DHT")

def adafruit_dht22_getter(pins=(4,14,17,24)):
    """
    Read out the dht22 sensor with help of the Adafruit read out
    software. Check data for glitches and filter out nonsensical 
    values.

    Keyword Args:
        pinst (tuple): The GPIO pins where the data line of the dht sensor
                       is connected to. Typically, a Raspberry Pi can support
                       up to 4 of these sensors.
    Returns:
        str - if success, None if failure

    """

    if not HAS_DHT:
        raise NotImplementedError("The dht22 sensor readout is not implemented because Adafruit_DHT is not installed on this system!")

    datastring = time.ctime() + "\t"
    SENSOR=Adafruit_DHT.AM2302
    try:
        for pin in pins:
            humidity, temperature = Adafruit_DHT.read(SENSOR, pin)
            if humidity is None: humidity = 101.00
            if temperature is None: temperature = -300.00
            datastring += "{:4.2f}\t{:4.2f}\t".format(temperature, humidity)
            time.sleep(0.2)
    except Exception as e:
        Logger.warning(f"Readout of the sensor on pin {pin} failed with Exception {e}")
        return None

    datastring += "\n"
    return datastring
