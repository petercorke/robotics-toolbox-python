import swift  # instantiate 3D browser-based visualizer
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np

env = swift.Swift()
env.launch(realtime=True)  # activate it

robot = rtb.models.Panda()
robot.q = robot.qr

T = SE3(0.5, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ikine_LM(T)  # solve IK
q_pickup = sol.q
qt = rtb.jtraj(robot.qr, q_pickup, 50)

env.add(robot)  # add robot to the 3D scene
for qk in qt.q:  # for each joint configuration on trajectory
    robot.q = qk  # update the robot state
    # robot.q = robot.qr
    env.step(0.05)  # update visualization


env.hold()
