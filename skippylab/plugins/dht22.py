import Adafruit_DHT
import datetime
import time

def adafruit_dht22_getter(pins=[4,14,17,24]):

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
        print (e)
        return None

    datastring += "\n"
    return datastring
