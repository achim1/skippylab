import socket, time

import skippylab.scpi.commands as cmd

import xdrlib
import vxi11.rpc as rpc

DEVICE_CORE_PROG  = 0x0607af
DEVICE_CORE_VERS  = 1

import vxi11.rpc as rpc



def make_header():
    # some constants (
    RPC_CALL = 1
    RPC_VERSION = 2
    
    MY_PROGRAM_ID = 1234 # assigned by Sun
    MY_VERSION_ID = 1000
    MY_TIME_PROCEDURE_ID = 9999
    
    AUTH_NULL = 0
    
    transaction = 1
    
    p = xdrlib.Packer()
    
    # send an Sun RPC call package
    p.pack_uint(transaction)
    p.pack_enum(RPC_CALL)
    p.pack_uint(RPC_VERSION)
    p.pack_uint(MY_PROGRAM_ID)
    p.pack_uint(MY_VERSION_ID)
    p.pack_uint(MY_TIME_PROCEDURE_ID)
    p.pack_enum(AUTH_NULL)
    p.pack_uint(0)
    p.pack_enum(AUTH_NULL)
    p.pack_uint(0)
    return p


class DynamicTCPTestServer(rpc.TCPServer):
    pass
    #def handl

class TestServer(object):

    def __init__(self):
        """
        A server which answers to vxi11.Instrument.ask
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 111))
        self.socket.listen()

    def __del__(self):
        self.socket.close()

    def recv(self):
        self.socket.recv(256)

    def serve(self):
        connected = False
        while not connected:
            conn, addr = self.socket.accept()
            connected = True
            time.sleep(1)
        while True:
            data = conn.recv(9198)
            print (data)
            print ("..")
            head = make_header()
            pack = rpc.Packer()
            pack.pack_replyheader(0,(0,b""))
            conn.send(pack.get_buf())
            while True:
                time.sleep(.5)
                conn.send(data)
            #if cmd.WHOAMI in data.decode():
            #    conn.send(b"<The mighty test server>")
            #print (conn, addr)
            #print (dir(conn))
            #time.sleep(20)

#dpwr = TestServer()
#dpwr.serve()
test = rpc.TCPServer("", 100000, 2, 111 )
test.loop()
    

