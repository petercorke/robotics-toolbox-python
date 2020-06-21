"""
NOT A UNIT TEST
Simple Example for Serial_link plot function
Starts by creating a robot arm.
uncomment and copy code to console as needed
"""

from roboticstoolbox.robot.serial_link import *

L = []

# create links
L.append(Link('a', 0.1, 'd', 1, 'alpha', pi/2, 'type', 'revolute'))
L.append(Link('a', 1, 'type', 'revolute'))
L.append(Link('a', 0.5, 'type', 'revolute'))

# create initial joint array to be passed into plot as joint configuration
qz = [pi/4,0,-pi/3]

# create serial link robot
arm = SerialLink(L, name='Upright Arm')

# plot robot
plotbot = arm.plot(qz)

## Use this code to change the joint angles
## lift arm
# plotbot.set_joint_angle(1, -pi/2)

## rotate base:
# plotbot.set_joint_angle(0, pi/2)

