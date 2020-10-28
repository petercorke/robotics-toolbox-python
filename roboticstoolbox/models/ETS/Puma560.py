from roboticstoolbox import ETS as ET
from roboticstoolbox import ERobot
l1 = 0.672
l2 = 0.2337
l3 = 0.4318
l4 = -0.0837
l5 = 0.4318
l6 = 0.0203

e = ET.tz(l1) * ET.rz() * ET.ty(l2) * ET.ry() * ET.tz(l3) * ET.tx(l6) * \
    ET.ty(l4) * ET.ry() * ET.tz(l5) * ET.rz() * ET.ry() * ET.rz() * ET.tx(0.2)

robot = ERobot(e, name="my first ERobot")
print(robot)
