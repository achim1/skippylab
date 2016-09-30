import osci, daq, plotting, tools, commands as cmd
import numpy as np

odaq = daq.DAQ("169.254.67.106",4000)
print (odaq.scope.WAITTIME)
odaq.find_best_acquisition_window(waveforms=20, trailing=100)
odaq.show_waveforms()
odaq.wf_header
wf = odaq.make_n_acquisitions(50000,trials = 100000, extra_timeout=0., \
                              return_only_charge = True)
np.save("measurement_ctrl_vlt",wf)
