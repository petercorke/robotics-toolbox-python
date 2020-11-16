#!/usr/bin/env python


import time
from pathlib import Path
import dynamixel_sdk as sdk                  # Uses Dynamixel SDK library

# list of all models https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_workbench/

# AX entry level
# XL improved control
# XM, XH better control, XH has Maxon motor
import json

class DynamixelString:

    def __init__(self, baudrate=1000000, port=None, protocol=1, top=256, verbose=True):

        jsonfile = Path(__file__).parent / "dynamixel.json"
        print(jsonfile)
        # load the JSON file and convert keys from strings to ints
        with jsonfile.open() as f:
            self.modeldict = {}
            for key, value in json.load(f).items():
                self.modeldict[int(key)] = value

        # setup the serial port
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.port = sdk.PortHandler(port)

        self.verbose = verbose

        # Open port
        if self.port.openPort():
            print("Succeeded to open the port")
        else:
            print("unable to open serial port", port)
            self.port = None
            raise RuntimeError("Failed to open the port")

        # Set port baudrate
        if self.port.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
        else:
            raise RuntimeError("Failed to change the baudrate")

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        if protocol == 1:
            self.packetHandler = sdk.PacketHandler("1.0")
        else:
            self.packetHandler = sdk.PacketHandler("2.0")

        # X-series memory map
        self.address = {}
        self.address['model'] = (2, 4)
        self.address['firmware'] = (6, 1)
        self.address['return_delay'] = (9, 1)
        self.address['drivemode'] = (10, 1, self._drivemode_handler)
        self.address['opmode'] = (11, 1, self._opmode_handler)
        self.address['shadow_id'] = (12, 1)

        self.address['min_position'] = (52, 4)
        self.address['max_position'] = (48, 4)
        self.address['velocity_limit'] = (44, 4)

        self.address['torque_enable'] = (64, 1)
        self.address['hardware_error_status'] = (70, 1)

        self.address['goal_position'] = (116, 4)
        self.address['goal_velocity'] = (104, 4)
        self.address['goal_pwm'] = (100, 2)

        self.address['present_position'] = (132, 4)
        self.address['present_position2'] = (132, 2)
        self.address['present_current'] = (126, 2)
        self.address['present_velocity'] = (128, 2)

        self.address['profile_velocity'] = (112, 4)
        self.address['profile_acceleration'] = (108, 4)

        self.address['temp'] = (146, 1)
        self.address['led'] = (65, 1)
        

        self.idlist = []
        self.models = {}

        # poll the ports
        for id in range(0, top):
            model = self.ping(id)
            if model > 0:
                self.idlist.append(id)
                self.models[id] = model
                self.set(id, 'torque_enable', False)
                self.set(id, 'led', False)
        print('servos found', self.idlist)

        error = self.get('all', 'hardware_error_status')
        if any(error):
            print('hardware error', error)

    def close(self):
        self.port.closePort()

    @property
    def n(self):
        return len(self.idlist)

    def ping(self, id):
        model_number, result, error = self.packetHandler.ping(self.port, id)
        return model_number

    def modelname(self, id):
        mn = self.models[id]
        try:
            return self.modeldict[mn]['model']
        except KeyError:
            return "?? mn={}".format(mn)

    def list(self):
        for id in self.idlist:
            fw = self.get(id, 'firmware')
            print("{:2d} {:3d} {:s}".format(id, fw, self.modelname(id)))

    def enable(self, on, id='all'):
        # enable(True)  # all enabled
        # enable (True, 1) # 1 is enabled
        # enable(True, [1, 2])  # 1, 2 are enabled
        # enable([True, False], [1, 2])  # 1 is enabled, 2 is disabled
        # enable([True, True, False, False]) # 1,2 enabled, 3,4 disabled
        if id == 'all':
            id = self.idlist
        if isinstance(id, list):
            if isinstance(on, bool):
                for id in self.idlist:
                    self.register_write(id, 'torque_enable', on)
            else:
                assert len(on) == len(id), 'incorrect enable flags'
                for i in id:
                    assert id in self.idlist, 'invalid id'
                    self.register_write(id, 'torque_enable', on[i])
        else:
            assert id in self.idlist, 'invalid id'
            self.register_write(id, 'torque_enable', on)


    def ismoving(self, id):
        return 0
    
    def setposition(self, id, pos):
        if id is None:
            # TODO allow numpy array, use argcheck.getvector
            assert isinstance(pos, (int, float)) or instance(pos, list) and len(pos) == self.n, 'incorrect position value'
            for id in self.idlist:
                self.register_write(id, 'goal_position', pos[i])
        else:
            assert id in self.idlist, 'invalid id'
            self.register_write(id, 'goal_position', pos)


    def getposition(self, id='all'):
        return self._getvalue('present_position', id)


    def _opmode_handler(self, op, id, register, value=None):
        dict = {1: "velocity", 3: "position", 4:"xposition", 16:"pwm"}

        if op == 'get':
            x = self.register_read(id, register)
            return dict[x]
        elif op == 'set':
            for k, v in dict.items():
                if v == value:
                     self.register_write(id, register, k)

    # for setting: handler(self, 'set', id, register, newvalue) -> value to write to register
    # for getting: handler(self, 'get', id, register) -> value to return
    def _drivemode_handler(self, op, id, register, value=None):
        if op == 'get':
            x = self.register_read(id, register)
            r = []
            if (x & 1) == 1:
                r.append('reverse')
            elif (x & 1) == 0:
                r.append('forward')
            elif (x & 4) == 4:
                r.append('tprofile')
            elif (x & 4) == 0:
                r.append('vprofile')
            return r
        elif op == 'set':
            curval = self.register_read(id, register)
            if value == 'reverse':
                curval |= 1
            elif value == 'forward':
                curval &= ~1
            elif value == 'tprofile':
                curval |= 4
            elif value == 'vprofile':
                curval &= ~4
            self.register_write(id, register, curval)

    # simple i/o using separate servo transations
    def get(self, id, register):
        if len(self.address[register]) > 2:
            reader = lambda id, register: self.address[register][2]('get', id, register)
        else:
            reader = self.register_read
        if id == 'all':
            id = self.idlist

        if isinstance(id, list):
            value = []
            for i in id:
                assert i in self.idlist, 'invalid id'
                value.append(reader(i, register))
            return value
        else:
            assert id in self.idlist, 'invalid id'
            return reader(id, register)

    def set(self, id, register, value):
        if self.verbose:
            print(f"set id={id}; {register} := {value}")

        if len(self.address[register]) > 2:
            writer = lambda id, register, value: self.address[register][2]('set', id, register, value)
        else:
            writer = self.register_write
        if id == 'all':
            id = self.idlist
        if isinstance(id, list):
            if isinstance(value, list):
                assert len(value) == len(id), 'length of value list must match number of servos'
                for i, v in zip(id, value):
                    assert i in self.idlist, 'invalid id'
                    writer(i, register, v)
            else:
                for i in id:
                    writer(i, register, value)
        else:
            assert id in self.idlist, 'invalid id %d' % (id,)
            writer(id, register, value)
    
    # wrapper on the Dynamixel SDK
    def register_read(self, id, regname):
        address, nbytes = self.address[regname][0:2]
        if nbytes == 1:
            dxl_value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.port, id, address)
        elif nbytes == 2:
            dxl_value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.port, id, address)
        elif nbytes == 4:
            dxl_value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.port, id, address)

        if dxl_comm_result != sdk.COMM_SUCCESS:
            raise RuntimeError(self.packetHandler.getTxRxResult(dxl_comm_result) + " id={}, register={}".format(id, regname))
        elif dxl_error != 0:
            raise RuntimeError(self.packetHandler.getRxPacketError(dxl_error) + " id={}, register={}".format(id, regname))
        return dxl_value


    def register_write(self, id, regname, value):
        #print('register_write', id, regname, value)
        if isinstance(regname, str):
            address, nbytes = self.address[regname][0:2]
        elif isinstance(regname, tuple) and len(regname) == 2:
            address, nbytes = regname
        else:
            raise ValueError('register must be a string or (address, length) tuple')
        if nbytes == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.port, id, address, value)
        elif nbytes == 2:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.port, id, address, value)
        elif nbytes == 4:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.port, id, address, value)

        if dxl_comm_result != sdk.COMM_SUCCESS:
            raise RuntimeError(self.packetHandler.getTxRxResult(dxl_comm_result) + " id={}, register={}".format(id, regname))
        elif dxl_error != 0:
            raise RuntimeError(self.packetHandler.getRxPacketError(dxl_error) + " id={}, register={}".format(id, regname))

    def register_indirect_config(self, id, indirectbase, regnames):
        iaddr = (indirectbase - 1) * 2 + 168
        count = 0

        for regname in regnames:
            address, nbytes = self.address[regname]
            for i in range(0, nbytes):
                self.register_write(id, (iaddr+i*2, 2), address+i)
                print(id, address+i, ' --> ', iaddr+i*2)

            iaddr += 2 * nbytes
            count += nbytes

        return ((indirectbase - 1) + 224, count)

    def register_read_sync(self, id, addrlen):
        # create an indirect table, keep track of how many bytes to be read
        groupSyncRead = GroupSyncRead(self.port, self.packetHandler, addrlen[0], addrlen[1])
        for id in self.active:
            dxl_addparam_result = groupSyncRead.addParam(id)
            if dxl_addparam_result != True:
                pass # bad thing

    def pulseleds(self, nblinks=4, dt=0.1):
        for i in range(0, nblinks):
            self.set('all', 'led', False)
            time.sleep(dt)
            self.set('all', 'led', True)
            time.sleep(dt)

class DynRobot:

    def __init__(self, dynamixels, arm, gripper):
        self.dynamixels = dynamixels
        self.idlist = []

        for motor in arm:
            print(motor)
            if isinstance(motor, int):
                self.idlist.append(motor)
            elif isinstance(motor, (tuple, list)) and len(motor) == 2:
                self.idlist.append(motor[0])
                if motor[1] < 0:
                    # motor is reveresed
                    second = -motor[1]
                    self.dynamixels.set(second, 'opmode', 'reverse')
                else:
                    second = motor[1]
                # second motor shadows the first
                self.dynamixels.set(second, 'shadow_id', motor[0])

        self.q0 = self.dynamixels.get(self.idlist, 'present_position')

    class SyncIO:
        pass

    def syncread_config(self, regnames, indirectbase=1):
        sr = SyncIO()   # lgtm [py/unused-local-variable]

        for id in self.idlist:
            (a, l) = d.register_indirect_config(id, indirectbase, regnames)     # lgtm [py/unused-local-variable]

    def setmode(self, m):
        self.dynamixels.set(self.idlist, 'drivemode', m)

    def moveto(self, q, t=2000, tacc=200, block=True):
        self.dynamixels.set(self.idlist, 'profile_velocity', int(t * 1000))
        self.dynamixels.set(self.idlist, 'profile_acceleration', int(tacc * 1000))
        self.dynamixels.set(self.idlist, 'goal_position', q)

    def getpos(self):
        pass

    def syncwrite_config(self):
        pass

    def enable(self, on):
        if on:
            self.dynamixels.pulseleds()
        self.dynamixels.set(self.idlist, 'torque_enable', on)
        if not on:
            self.dynamixels.set('all', 'led', False)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Dynamixel interactive console')
    parser.add_argument('--port', '-p', default="/dev/tty.usbserial-FT4NQ6ZP",
        help='specify name of serial port')
    parser.add_argument('--baudrate', '-b', type=int, default=1000000,
        help='serial baud rate')
    parser.add_argument('--protocol', type=int, default=2,
        help='Dynamixel protocol level')
    parser.add_argument('--scan', type=int, default=20,
        help='Scan for Dynamixel ids 1 to this number inclusive')
    args = parser.parse_args()

    dms = DynamixelString(port=args.port,
         baudrate=args.baudrate, 
         protocol=args.protocol,
         top=args.scan)
    dms.set('all', 'return_delay', 0)

    # simple console-based command handler
    while True:
        cmd = input("dynamixel>>> ")
        cmd = cmd.strip()
        cmd = cmd.split(",")
        if cmd[0] == "list":
            dms.list()
        elif cmd[0] == "read":
            p = d.getposition()
            print(p)
        elif cmd[0] == "limp":
            dms.enable(False)
        elif cmd[0] == "hold":
            dms.enable(True)
        elif cmd[0] == "led":
            status = int(cmd[1])
            dms.setled(status > 0)
        elif cmd[0] == "temp":
            print(dms.gettemp())
        elif cmd[0] == "quit" or cmd[0] == "exit":
            break
