#! /usr/bin/env python3

import numpy as np
import hjson
from skippylab.instruments import RaspberryPiGPIODHT22Thermometer, RaspberryPiGPIODHT22ThermometerSingleChannel 
from skippylab.controllers import ZMQController

savetodb = False
try:
    import django
    django.setup()
    from monitoring.models import FreezerTemperatureRecord as freezy
    savetodb = True
except Exception as e:
    print (e) 
    print ("Can not import django")

def interpret_payload(payload):
    """
    Convert the strings to types

    """
    for k in RaspberryPiGPIODHT22Thermometer.TYPES:
        payload[k] = RaspberryPiGPIODHT22Thermometer.TYPES[k](payload[k])
    return payload


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Recieve termperature data monitored elsewhere")
    parser.add_argument("-i","--ip", help="Subscribe to instrument at ip",
                        type=str, default="168.105.246.157")
    parser.add_argument("-t","--topic", help="Topicfilter -  filter messages by type",
                        type=str, default="FREEZER")
    parser.add_argument("--save-to-db", help="Save data to mongo db",
                        action='store_true', default=False)
    parser.add_argument("--debug", help="Turn on debug flags",
                        action='store_true', default=False)
    parser.add_argument("-p","--port", help="Subscribe to port at ip",
                        type=int, default=9876)
    parser.add_argument("-c","--channels", help="How many channels does the thermometer/temperature sensor have?",
                        type=int, default=4)
    args = parser.parse_args()

    print (args)

    if args.channels == 1:
        controller = ZMQController(ip=args.ip,topicfilter=args.topic,
                                   encoder=lambda x : RaspberryPiGPIODHT22Thermometer.encode_payload(x, reverse_string_template=RaspberryPiGPIODHT22ThermometerSingleChannel.REVERSE_STRING_TEMPLATE))
        thermo = RaspberryPiGPIODHT22ThermometerSingleChannel(controller=controller)

    else:
        controller = ZMQController(ip=args.ip,topicfilter=args.topic,
                               encoder=lambda x : RaspberryPiGPIODHT22Thermometer.encode_payload(x, reverse_string_template=RaspberryPiGPIODHT22Thermometer.REVERSE_STRING_TEMPLATE))
        thermo = RaspberryPiGPIODHT22Thermometer(controller=controller)
    for payload in thermo.measure_continuously(measurement_time=np.inf):
        if args.save_to_db:
            payload = hjson.loads(payload)
            #payload = RaspberryPiGPIODHT22Thermometer.TYPES
            payload = interpret_payload(payload)
            data = freezy()
            data.from_payload(payload)
            data.save()
        print (payload)


