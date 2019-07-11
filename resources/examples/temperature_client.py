#! /usr/bin/env python3

import numpy as np
from skippylab.instruments import RaspberryPiGPIODHT22Thermometer
from skippylab.controllers import ZMQController

FILE = 'chamber_second_run.dat'

if __name__ == "__main__":

    controller = ZMQController(topicfilter="FREEZER", encoder=RaspberryPiGPIODHT22Thermometer.encode_payload)
    thermo = RaspberryPiGPIODHT22Thermometer(controller=controller)
    f = open(FILE, 'w')
    for k in thermo.measure_continuously():
        print (k)
        f.write(k)
