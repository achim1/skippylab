"""
Plugin function to read out a sensirion sh31 temperature/humidity sensor
The sensor connects to the pi via the ic2 bus
"""
import time
import sys 
import datetime
import dateutil.tz as tz
import os

# set the timezone to utc
os.environ['TZ'] = 'UTC'
time.tzset()

try:
    import board
    import busio
    import adafruit_sht31d
except ModuleNotFoundError:
    print ("Can not use adafruit dht sensors on this machine. Try installing requirements with pip3 install adafruit-circuitpython-sht31d")


def sht31d_getter():
    """
    Query the sensor
    """

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_sht31d.SHT31D(i2c)

    datastring = time.ctime() + "\t"
    retries = 5
    while True:
        if not retries:
            humidity = np.nan
            temperature = np.nan
            break
        try:
            humidity = sensor.relative_humidity
            temperature = sensor.temperature
            break
        except:
            time.sleep(1)
            retries -= 1
            continue

    datastring += "{:4.2f}\t{:4.2f}\n".format(temperature, humidity)
    return datastring

if __name__ == '__main__':
    
    while True:
        print (sht31d_getter())
        time.sleep(2)
