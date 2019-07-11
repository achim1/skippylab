#! /usr/bin/env python3

import numpy as np
from skippylab.instruments import RaspberryPiGPIODHT22Thermometer, RaspberryPiGPIODHT22ThermometerSingleChannel
from skippylab.controllers import GPIOController
from skippylab.plugins.dht22 import adafruit_dht22_getter


if __name__ == "__main__":

    #controller = GPIOController(data_getter=adafruit_dht22_getter,\
    #                            data_getter_kwargs = {"pins" : [4,14,17,24] })
    controller = GPIOController(data_getter=adafruit_dht22_getter,\
                                data_getter_kwargs = {"pins" : [4] })
    thermo = RaspberryPiGPIODHT22ThermometerSingleChannel(controller=controller,\
                                             publish=True)
    for k in thermo.measure_continuously(measurement_time=np.inf, interval=10):
        print (k)

