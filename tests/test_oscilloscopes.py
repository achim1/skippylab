import pytest

#
# As port 111 is privileged, unfortunatley this requires root privileges
#

from skippylab.instruments.oscilloscopes import UnknownOscilloscope

def test_UnknownOscilloscope():
    scope = UnknownOscilloscope("127.0.0.1")
    scope._send(1)
    scope._send("hello")
    
    pingable = scope.ping()
    assert pingable

