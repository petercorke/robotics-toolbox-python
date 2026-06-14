r = DynRobot(dms, [1, (2, -3), (4, -5), 6, 7, 8], 9)
r.setmode("position")
r.enable(True)

# r.moveto([0.0]*6, t=5000)

# time.sleep(5)


# r.syncread_config(1, ['present_position2', 'present_current'])

# print(d.get(1, 'firmware'))
# q0 = d.get('all', 'present_position')
# print(q0, type(q0))


# d.set('all', 'torque_enable', False)

# d.set('all', 'opmode', 'position')
# d.set(5, 'shadow_id', 4)   # 4 shadows 5
# d.set(3, 'shadow_id', 2)   # 3 shadows 2
# d.set('all', 'opmode', 'forward')
# d.set([3, 5], 'opmode', 'reverse')
# d.set('all', 'velocity_limit', 2)

# d.set('all', 'profile_velocity', 5000)
# d.set('all', 'profile_acceleration', 500)


# joint   servo
#  1       1
#  2       2 -3
#  3       4 -5
#  4       6
#  5       7
#  6       8
#  G       9

dms.list()

# d.set('all', 'torque_enable', True)

# d.set([1, 2, 4, 6, 7, 8, 9], 'goal_position', 2000)
# time.sleep(10)


# for i in range(0,10):
#     time.sleep(1)
#     p =d._getvalue('present_position2', 4)
#     print(p)


for id in dms.idlist:
    (a, l) = dms.register_indirect_config(
        id, 1, ("present_position2", "present_current")
    )

group_read = sdk.GroupSyncRead(dms.port, dms.packetHandler, a, l)
for id in [1, 2, 4, 6, 7, 8, 9]:
    dxl_addparam_result = group_read.addParam(id)
    if dxl_addparam_result != True:
        raise RuntimeError("groupSyncRead failed to add id=%d" % id)

t0 = time.time()

for i in range(0, 1000):
    dxl_comm_result = group_read.txRxPacket()
    if dxl_comm_result != sdk.COMM_SUCCESS:
        raise RuntimeError("%s" % dms.packetHandler.getTxRxResult(dxl_comm_result))

    time.sleep(0.1)

# for id in [1, 2, 4, 6, 7, 8, 9]:
#     dxl_getdata_result = group_read.isAvailable(id, a, l)
#     if dxl_getdata_result != True:
#         print("[ID:%03d] groupSyncRead getdata failed" % DXL2_ID)

t1 = time.time()
print("reading time ", (t1 - t0) / 100)

for id in [1, 2, 4, 6, 7, 8, 9]:
    position = group_read.getData(id, a, 2)
    current = group_read.getData(id, a + 2, 2)
    if current & 0x8000 > 0:
        current = current - 0x10000
    print(id, ": ", position, current)


# d.set([1, 2, 4, 6, 7, 8, 9], 'goal_position', [q0[i-1] for i in [1, 2, 4, 6, 7, 8, 9]])

# for i in range(0, 5):
#     for id in d.idlist:
#         d.setled(True, id)
#         time.sleep(0.5)
#         d.setled(False, id)


# print(d.gettemp())
# d.setled(True)
# # d.enable(True)
# p = d.getposition()
# print(p)
# time.sleep(5)
# d.setled(False)
# p = d.getposition()
# print(p)
# d.enable(False)

time.sleep(10)
