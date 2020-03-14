#! /usr/bin/env python3

import sys
import numpy as np

import skippylab.instruments as instr
import skippylab.plugins.sht31 as s31


if __name__ == '__main__':
    therm = instr.thermometers.RPIMultiChannelThermometerServer(data_getters=(s31.sht31d_getter,), topics=('DRYROOM',))

    outfile = open(sys.argv[1],'a')

    for data in therm.measure_continuously(measurement_time=np.inf, interval=2):
        print (data['DRYROOM'])
        outfile.write(data['DRYROOM'])
        outfile.flush()

