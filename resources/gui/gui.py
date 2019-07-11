"""
Graphical user interface with the tkinter package to visualize data
readout from a Milli Gauss instrument
"""


import tkinter, time
import tkinter.messagebox
import numpy as np
import pylab as p
import hjson
import seaborn as sb

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from skippylab import __version__, get_logger#, create_timestamped_file

class InstrumentGui(object):
    """
    Simple gui plotting solution for any instrument

    """

    def __init__(self, master, instrument, interval=5,\
                 maxpoints=200, loglevel=20):
        """
        Initialize the application window

        Args:
            master (tkinter.Tk): A tkinter main application

        Keyword Args:
            interval (int): Update the plot every interval seconds
            maxpoints (int): Max number of points visible in the plot


        """
        # Create container and menus
        self.master = master
        self.logger = get_logger(loglevel)
        self.frame = tkinter.Frame(self.master)
        top = self.master.winfo_toplevel()
        self.menu_bar = tkinter.Menu(top)
        top['menu'] = self.menu_bar

        self.sub_menu_help = tkinter.Menu(self.menu_bar)
        self.sub_menu_plot = tkinter.Menu(self.menu_bar)
        self.menu_bar.add_cascade(label='Plot', menu=self.sub_menu_plot)
        self.menu_bar.add_cascade(label='Help', menu=self.sub_menu_help)
        self.sub_menu_help.add_command(label='About', command=self._about_handler)
        self.sub_menu_plot.add_command(label="Reset", command=self.init_plot)
        self.sub_menu_plot.add_command(label="Log to file", command=self.init_datafile)

        # physics quantities
        self.instrument = instrument
        self.start_time = time.monotonic()
        self.interval = interval
        self.maxpoints = maxpoints

        # writing results to file
        self.datafile_active = False
        self.datafilename = None
        self.datafile = None

        # get to know some things about the instrument
        identifier = self.instrument.identify()
        self.logger.info("Found instrument {}".format(identifier))

        # plot
        fig = Figure()
        self.ax = fig.gca()
    
        self.tinax = False
        self.twinline = False
        if identifier["twinax"]:
            self.twinax = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        try:
            self.canvas.show()
        except AttributeError:
            self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.frame.pack()
        self.init_plot()
        self.update()
       
    @staticmethod
    def _about_handler():
        """
        Action performed if "about" menu item is clicked

        """
        tkinter.messagebox.showinfo("About", "Version: {}".format(__version__))

    def init_plot(self):
        """
        Initialize the plot
        """

        #unit = self.instrument.unit
        unit = "C"
        axis_label = self.instrument.axis_label
        self.ax.set_xlabel("measurement time [s]")
        self.ax.set_ylabel("{} [{}]".format(axis_label, unit))

        identifier = self.instrument.identify()
        self.lines = []
        self.twinlines = []
        for label in identifier["channels"]:
            line, = self.ax.plot_date(range(0), [], color="red", lw=3, label=label)
            self.lines.append(line)
        if self.twinax:
            for __ in identifier["channels"]:
                line, = self.twinax.plot_date(range(0), [], color="blue", lw=3, label=label)
                self.twinlines.append(line)

    def init_datafile(self):
        """
        Write measurement results to a logfile
        """
        if self.datafile_active:
            self.datafile.close()

        #self.datafilename = create_timestamped_file("GAUSSMETER_GU3001D_", file_ending=".dat")
        self.datafilename  = "fixme"
        self.logger.info("Writing to file {}".format(self.datafilename))
        tkinter.messagebox.showinfo("Writing to a file!", "Writing data to file {}".format(self.datafilename))
        self.datafile = open(self.datafilename, "w")
        self.datafile.write("# seconds {}\n".format(self.instrument.unit))
        self.datafile_active = True

    def update(self):
        """
        Update the plot with recent magnetoinstrument data
        """
        meta = self.instrument.identify()
        try:
            payload = self.instrument.measure()
        except Exception as e:
            self.logger.warning("Can not acquire data! {}".format(e))
            return 
            
        payload = hjson.loads(payload)
        for k in payload:
            payload[k] = self.instrument.TYPES[k](payload[k])

        print (type(payload))


        index = 0
        timestamps,__ = self.lines[0].get_data()
        timestamp = payload[meta["xdata"]]
        if len(timestamps) >= self.maxpoints:
            self.logger.debug("Restricting line to {} points".format(self.maxpoints))
            index = 1
        timestamps = np.append(timestamps[index:], timestamp)

        minydataax1 = np.inf
        maxydataax1 = -np.inf
        minydataax2 = np.inf
        maxydataax2 = -np.inf

        for i,line in enumerate(self.lines):
            __, line_ydata = line.get_data()
            index = 0
            if len(line_ydata) >= self.maxpoints:
                self.logger.debug("Restricting line to {} points".format(self.maxpoints))
                index = 1
            newdata = payload[meta["channels"][i]]
            line_ydata = np.append(line_ydata[index:], newdata)
            if min(line_ydata) < minydataax1: 
                minydataax1 = min(line_ydata)
            if max(line_ydata) > maxydataax1:
                maxydataax1 = max(line_ydata)
            line.set_ydata(line_ydata)
            
            line.set_xdata(timestamps)

        for i,line in enumerate(self.twinlines):
            __, line_ydata = line.get_data()
            index = 0
            if len(line_ydata) >= self.maxpoints:
                self.logger.debug("Restricting line to {} points".format(self.maxpoints))
                index = 1
            newdata = payload[meta["twinaxchannels"][i]]
            line_ydata = np.append(line_ydata[index:], newdata)
            if min(line_ydata) < minydataax2: 
                minydataax2 = min(line_ydata)
            if max(line_ydata) > maxydataax2:
                maxydataax2 = max(line_ydata)
            

            line.set_ydata(line_ydata)
            line.set_xdata(timestamps)
            print (timestamps)
            print (line_ydata)
        #if len(timestamps) > 5: # avoid itcks exceed blabla error
        #    if "xmaj_formatter" in meta:
        #        self.ax.xaxis.set_major_formatter(meta["xmaj_formatter"])
        #    if "xmaj_locator" in meta:
        #        self.ax.xaxis.set_major_locator(meta["xmaj_locator"])
        #    if "xmin_formatter" in meta:
        #        self.ax.xaxis.set_minor_formatter(meta["xmin_formatter"])
        #    if "xmin_locator" in meta:
        #        self.ax.xaxis.set_minor_locator(meta["xmin_locator"])
        #    #if self.twinax:
        #    #    if "xmaj_formatter" in meta:
        #    #        self.twinax.xaxis.set_major_formatter(meta["xmaj_formatter"])
        #    #    if "xmaj_locator" in meta:
        #    #        self.twinax.xaxis.set_major_locator(meta["xmaj_locator"])
        #    #    if "xmin_formatter" in meta:
        #    #        self.twinax.xaxis.set_minor_formatter(meta["xmin_formatter"])
        #    #    if "xmin_locator" in meta:
        #    #        self.twinax.xaxis.set_minor_locator(meta["xmin_locator"])
        #for line in self.twinlines:
        #    timestamps, fields = line.get_data()    

        #sec = time.monotonic() - self.start_time

        ## make sure data in the plot is "falling over"
        ## so that it does not get too crammed
        #index = 0
        #if len(secs) >= self.maxpoints:
        #    self.logger.debug("Restricting line to {} points".format(self.maxpoints))
        #    index = 1

        #secs = np.append(secs[index:], sec)
        #if payload is not None:
        #    payload = np.append(fields[index:], field)

        #datamin = min(fields)
        #datamax = max(fields)
        xmin = min(timestamps)
        xmax = max(timestamps)
        
        ## avoid matplotlib warning
        #if abs(datamin - datamax) < 1:
        #    datamin -= 1
        #    datamax += 1

        #if abs(xmax - xmin) < 1:
        #    xmin -= 1
        #    xmax += 1

        ## write to the datafile if desired
        #if self.datafile_active:
        #    self.datafile.write("{:4.2f} {:4.2f}\n".format(sec, field))
        self.ax.set_xlim(xmin=xmin, xmax=xmax)
        print (minydataax1)
        print (maxydataax1)
        print (minydataax2)
        print (maxydataax2)

        if np.isfinite(minydataax1) and np.isfinite(maxydataax1):
            self.ax.set_ylim(ymin=minydataax1, ymax=maxydataax1)
        if np.isfinite(minydataax2) and np.isfinite(maxydataax2):
            self.twinax.set_ylim(ymin=minydataax2, ymax=maxydataax2)
        #self.line.set_ydata(fields)
        #self.line.set_xdata(secs)
        self.canvas.draw()
        self.master.after(self.interval, self.update)


#
#class LutronInstrumentGraphical(object):
#    """
#    A TKinter widget to visualize Gauss instrument data
#    """
#
#    def __init__(self, master, instrument,  interval=2,\
#                 maxpoints=200, loglevel=20):
#       ` """
#        Initialize the application window
#
#        Args:
#            master (tkinter.Tk): A tkinter main application
#
#        Keyword Args:
#            interval (int): Update the plot every interval seconds
#            maxpoints (int): Max number of points visible in the plot
#
#
#        """
#
#        # Create container and menus
#        self.master = master
#        self.logger = get_logger(loglevel)
#        self.frame = tkinter.Frame(self.master)
#        top = self.master.winfo_toplevel()
#        self.menu_bar = tkinter.Menu(top)
#        top['menu'] = self.menu_bar
#
#        self.sub_menu_help = tkinter.Menu(self.menu_bar)
#        self.sub_menu_plot = tkinter.Menu(self.menu_bar)
#        self.menu_bar.add_cascade(label='Plot', menu=self.sub_menu_plot)
#        self.menu_bar.add_cascade(label='Help', menu=self.sub_menu_help)
#        self.sub_menu_help.add_command(label='About', command=self._about_handler)
#        self.sub_menu_plot.add_command(label="Reset", command=self.init_plot)
#        self.sub_menu_plot.add_command(label="Log to file", command=self.init_datafile)
#
#        # physics quantities
#        self.instrument = meter
#        self.start_time = time.monotonic()
#        self.interval = interval
#        self.maxpoints = maxpoints
#
#        # writing results to file
#        self.datafile_active = False
#        self.datafilename = None
#        self.datafile = None
#
#        # plot
#        fig = Figure()
#        self.ax = fig.gca()
#        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
#        self.canvas.show()
#        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
#        self.frame.pack()
#        self.init_plot()
#        self.update()
#
#    def init_plot(self):
#        """
#        Initialize the plot
#        """
#
#        unit = self.instrument.unit
#        axis_label = self.instrument.axis_label
#        self.ax.set_xlabel("measurement time [s]")
#        self.ax.set_ylabel("{} [{}]".format(axis_label, unit))
#        self.line, = self.ax.plot(range(0), color="blue", lw=3)
#
#    @staticmethod
#    def _about_handler():
#        """
#        Action performed if "about" menu item is clicked
#
#        """
#        tkinter.messagebox.showinfo("About", "Version: {}".format(__version__))
#
#    def update(self):
#        """
#        Update the plot with recent magnetoinstrument data
#        """
#
#        secs, fields = self.line.get_data()
#        field = None
#        try:
#            field = self.instrument.measure()
#        except Exception as e:
#            self.logger.warning("Can not acquire data! {}".format(e))
#
#        sec = time.monotonic() - self.start_time
#
#        # make sure data in the plot is "falling over"
#        # so that it does not get too crammed
#        index = 0
#        if len(secs) >= self.maxpoints:
#            self.logger.debug("Restricting line to {} points".format(self.maxpoints))
#            index = 1
#
#        secs = np.append(secs[index:], sec)
#        if field is not None:
#            fields = np.append(fields[index:], field)
#
#        datamin = min(fields)
#        datamax = max(fields)
#        xmin = min(secs)
#        xmax = max(secs)
#
#        # avoid matplotlib warning
#        if abs(datamin - datamax) < 1:
#            datamin -= 1
#            datamax += 1
#
#        if abs(xmax - xmin) < 1:
#            xmin -= 1
#            xmax += 1
#
#        # write to the datafile if desired
#        if self.datafile_active:
#            self.datafile.write("{:4.2f} {:4.2f}\n".format(sec, field))
#        self.ax.set_xlim(xmin=xmin, xmax=xmax)
#        self.ax.set_ylim(ymin=datamin, ymax=datamax)
#        self.line.set_ydata(fields)
#        self.line.set_xdata(secs)
#        self.canvas.draw()
#        self.master.after(self.interval, self.update)
#
#    def init_datafile(self):
#        """
#        Write measurement results to a logfile
#        """
#        if self.datafile_active:
#            self.datafile.close()
#
#        self.datafilename = create_timestamped_file("GAUSSMETER_GU3001D_", file_ending=".dat")
#        self.logger.info("Writing to file {}".format(self.datafilename))
#        tkinter.messagebox.showinfo("Writing to a file!", "Writing data to file {}".format(self.datafilename))
#        self.datafile = open(self.datafilename, "w")
#        self.datafile.write("# seconds {}\n".format(self.instrument.unit))
#        self.datafile_active = True
#
#    def __del__(self):
#        """
#        Close open files
#        """
#        if self.datafile_active:
#            self.datafile.close()
#
#
