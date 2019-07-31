#pp = risingsun.chamber.PrologixUsbGPIBController(gpib_adress=7)
#print (pp.query('*IDN?'))
from enum import Enum

from .abstractbaseinstrument import AbstractBaseInstrument
from ..controllers import PrologixUsbGPIBController, NI_GPIB_USB

class CytecPatchPannelCommands:

    IDN              = "*IDN?"
    CURRENT_SETTINGS = "D"
    CLEAR            = "C"
    REVISION         = "N"
    STATUS           = "S"

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
    

cppc = CytecPatchPannelCommands

class Cytec(AbstractBaseInstrument):
    """
    A switching system from cytec
    """
    MODULES  = [0,1,2,3]
    SWITCHES = [0,1,2,3,4,5,6,7]
    ROWS     = [[7,3],[6,2],[5,1],[4,0]] # from top to bottom

    def __init__(self, controller, port=9999, publish=False):
    
        assert (isinstance(controller, NI_GPIB_USB) or isinstance(controller, PrologixUsbGPIBController)), "The use    d controller has to be either the NI usb one or the prologix usb"
    
        self._controller = controller
        self.publish = publish
        self.port = port
        self._socket = None

    def identify(self):
        return self._controller.query(cppc.IDN)

    def get_current_settings(self):
        return self._controller.query(cppc.CURRENT_SETTINGS)

    def unlatch_all(self):
        return CytecPatchPannelStatus(self._controller.query(cppc.CLEAR))

    def clear(self):
        return self.unlatch_all()

    @property
    def revision_number(self):
        return self._controller.query(cppc.REVISION)


    def show_matrix(self):
        matrix = self._controller.query(cppc.STATUS)
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
        return CytecPatchPannelStatus(self._controller.query('L {} {}'.format(module, switch)))

    def raw_unlatch(self, module, switch):
        return CytecPatchPannelStatus(self._controller.query('L {} {}'.format(module, switch)))
        
    def latch_all(self):
        for mod in self.MODULES:
            for sw in self.SWITCHES:
                self.raw_latch(mod, sw)

    def how_to_connect_gaps_detectormodules(self):
        explanation = 'Please connect quadrant 1 (det 0) in the uppermost row and below quadrant 2, 3 and 4 in such a way that 4 is the lowest!'
        return explanation

    def latch_detector(self, det):
        assert (det >= 0 and det < 4), "Detector numbers go from 0 to 3"
        
        # latch rows
        for mod in self.MODULES:
            for sw in self.ROWS[det]:
                self.raw_latch(mod, sw)


    
