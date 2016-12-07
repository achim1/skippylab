import osci, daq, plotting, tools, commands as cmd
import numpy as np

# initialize with scope IP adress and port
odaq = daq.DAQ("169.254.67.106",4000)

# find a good acquisiton window around peak in waveform
odaq.find_best_acquisition_window(waveforms=20, trailing=100)

# display some waveforms on the screen
odaq.show_waveforms()

# header stores information about x and y ticks
head = odaq.scope.get_wf_header()

# take actual data (e.g. 50000 waveforms) and save it to a 
# numpy compatible file 
wf = odaq.make_n_acquisitions(50000)
tools.save_waveform(head, wf, "measurement_ctrl_vlt")
