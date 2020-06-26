"""
Operate a multi switch module (patchpanel) manufactured by Cytec

"""
from enum import Enum
import tqdm

from .abstractbaseinstrument import AbstractBaseInstrument
from ..controllers import PrologixUsbGPIBController, SimpleSocketController

class CytecPatchPannelCommands:

    IDN              = "*IDN?"
    CURRENT_SETTINGS = "D"
    CLEAR            = "C"
    REVISION         = "N"
    STATUS           = "S"

class CytecConnectionType:
    ETHER            = 1
    GPIB             = 2
    UNKNOWN          = 10
# from the manual
#‘0’ Successful operation, switch open
#‘1’ Successful  operation, switch closed
#‘2’ Unknown command, the first character of the command string was unrecognizable
#‘3’ Incorrect entries, the number or type of entries was incorrect
#‘4’ Entries out of limits, a switch point was requested that was outside the limits of the specified matrix
#‘5’ Invalid access code, the code number 73 was not included or entered incorrectly
#‘6’ Setup error. Incorrect setup parameter

class CytecPatchPannelStatus(Enum):
    SUCCESS_OPEN    = '0'
    SUCCESS_CLOSED  = '1'
    UNKNOWN_CMD     = '2' # first character not recognized
    INCORRECT_ENT   = '3' # number of entries incorrect
    OUTOFLIMIT      = '4' 
    INVALIDACCESS   = '5'
    SETUP_ERR       = '6' 
    UNKNOWN_STAT    = '-1'    

cppc = CytecPatchPannelCommands

class Cytec(AbstractBaseInstrument):
    """
    A switching system from cytec
    """
    MODULES  = [0,1,2,3]
    SWITCHES = [0,1,2,3,4,5,6,7]
    ROWS     = [[7,3],[6,2],[5,1],[4,0]] # from top to bottom

    def __init__(self, controller):
        """
        Intitialize a connection to a patch panel from Cytec via a sckippylab.controler

        Args:
            controller () : Either a Telnet or GPIB controller

        """   
 
        #assert (isinstance(controller, NI_GPIB_USB) or isinstance(controller, PrologixUsbGPIBController)), "The use    d controller has to be either the NI usb one or the prologix usb"
    
        self._controller = controller
        self.connection_type = CytecConnectionType.UNKNOWN
        if (isinstance(controller, SimpleSocketController)): 
            self.connection_type = CytecConnectionType.ETHER
        elif (isinstance(controller, TelnetController)): 
            self.connection_type = CytecConnectionType.ETHER
        else:
            self.connection_type = CytecConnectionType.GPIB 

    def _parse(self, response):
        if self.connection_type == CytecConnectionType.ETHER:
            response = response.replace("\r\n",";")
        return response

    def identify(self):
        return self._controller.query(cppc.IDN)

    def get_current_settings(self):
        data = self._controller.query(cppc.CURRENT_SETTINGS)
        if self.connection_type == CytecConnectionType.ETHER:
            data += self._controller.read()
            data += self._controller.read()
            data += self._controller.read()
            data += self._controller.read()
            data += self._controller.read()
        data = self._parse(data)
        return data

    def unlatch_all(self):
        status = CytecPatchPannelStatus("-1")
        try:
            status = CytecPatchPannelStatus(self._controller.query(cppc.CLEAR))
        except:
            pass
        return status

    def clear(self):
        return self.unlatch_all()

    @property
    def revision_number(self):
        data = self._controller.query(cppc.REVISION)
        data = self._parse(data)
        return data

    def show_matrix(self):
        matrix = self._controller.query(cppc.STATUS)
        matrix = self._parse(matrix)
        matrix = matrix.split(';')
        return matrix

    @property
    def n_modules(self):
        return len(self.MODULES)

    @property
    def n_swtiches_per_module(self):
        return len(self.SWITCHES)

    def explain(self):
        print ('4 modules with 8 switches each')
        print ('switch 0-3 connects to output 0 of each module')
        print ('switch 4-7 connects to output 1 of each module')

    def raw_latch(self, module, switch):
        status = CytecPatchPannelStatus('-1')
        try:
            status =  CytecPatchPannelStatus(self._controller.query('L {} {}'.format(module, switch)))
        except:
            pass 
        return status

    def raw_unlatch(self, module, switch):
        status = CytecPatchPannelStatus("-1")
        try:
            status =  CytecPatchPannelStatus(self._controller.query('L {} {}'.format(module, switch)))
        except:
            pass
        return status        

    def latch_all(self):
        for mod in self.MODULES:
            for sw in self.SWITCHES:
                self.raw_latch(mod, sw)

    def how_to_connect_gaps_detectormodules(self):
        explanation = 'Please connect quadrant 1 (det 0) in the uppermost row and below quadrant 2, 3 and 4 in such a way that 4 is the lowest!'
        return explanation

    def latch_detector(self, det):
        """
        Latch a specific detector for a module used in the GAPS experiment
        
        Args:
            det (int) : detector number on the mudule (1-4)
        """

        assert (det >= 1 and det < 5), "Detector numbers go from 1 to 4"
        self.unlatch_all()        
        # latch rows
        for mod in tqdm.tqdm(self.MODULES):
            for sw in self.ROWS[det -1]:
                #print (f"Latching module {mod} switch {sw}")
                self.raw_latch(mod, sw)


    
