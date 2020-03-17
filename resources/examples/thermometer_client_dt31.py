#! /usr/bin/env python3

import time
import datetime
import dateutil.tz as tz
import pytz
import numpy as np
import hjson
import re 
from skippylab.controllers import ZMQController

savetodb = False
try:
    import django
    django.setup()
    from monitoring.models import HumidityTemperatureRecord as htrecord
    savetodb = True
except Exception as e:
    print (e) 
    print ("Can not import django")



PAYLOADREGEXP = re.compile("\s(?P<timestamp>[A-Za-z0-9\s:]*)\s(?P<temp>[0-9\.-]*)\s(?P<humi>[0-9\.]*)")

def encoder(payload):
    result = {}
    parsed = PAYLOADREGEXP.search(payload)
    if parsed is not None:
        #{'timestamp': '\tSun Mar 15 15:41:09 2020', 'temp': '27.52', 'humi': '34.64'}
	#Sun Mar 15 15:41:09 2020	27.52	34.64
        result['timestamp'] =  datetime.datetime.strptime(parsed['timestamp'],"%a %b %d %H:%M:%S %Y")
        result['timestamp'] = result['timestamp'].replace(tzinfo=pytz.timezone('UTC'))
        result['temp'] = float(parsed['temp'])
        result['humi'] = float(parsed['humi'])
    return result

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

    controller = ZMQController(ip=args.ip,topicfilter=args.topic,
                               encoder=encoder)


    while True:
        payload = controller.read()
        if payload:
            record = htrecord()
            record.topic = args.topic
            record.temperature = payload['temp']
            record.humidity = payload['humi']
            record.timestamp = payload['timestamp']
            record.save()
        print (payload)
        time.sleep(20)
#
#    if args.channels == 1:
#        controller = ZMQController(ip=args.ip,topicfilter=args.topic,
#                                   encoder=lambda x : RaspberryPiGPIODHT22Thermometer.encode_payload(x, reverse_string_template=RaspberryPiGPIODHT22ThermometerSingleChannel.REVERSE_STRING_TEMPLATE))
#        thermo = RaspberryPiGPIODHT22ThermometerSingleChannel(controller=controller)
#
#    else:
#        controller = ZMQController(ip=args.ip,topicfilter=args.topic,
#                               encoder=lambda x : RaspberryPiGPIODHT22Thermometer.encode_payload(x, reverse_string_template=RaspberryPiGPIODHT22Thermometer.REVERSE_STRING_TEMPLATE))
#        thermo = RaspberryPiGPIODHT22Thermometer(controller=controller)
#    for payload in thermo.measure_continuously(measurement_time=np.inf):
#        if args.save_to_db:
#            payload = hjson.loads(payload)
#            #payload = RaspberryPiGPIODHT22Thermometer.TYPES
#            payload = interpret_payload(payload, instrument=thermo)
#            data = freezy()
#            data.from_payload(payload)
#            data.topic = args.topic
#            data.save()
#
#        print (payload)
#
#
