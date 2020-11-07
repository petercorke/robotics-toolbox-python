# this is the example code from the t0p-level README..d
import roboticstoolbox as rtb

robot = rtb.models.DH.Panda()
print(robot)
T = robot.fkine(robot.qz)
print(T)
qt = rtb.trajectory.jtraj(robot.qz, robot.qr, 50)
robot.plot(qt.q, movie="panda1.gif")

# IK
from spatialmath import SE3

T = SE3(0.1, 0.2, 0.5) * SE3.OA([0, 1, 0], [0, 0, -1])
q, *_ = robot.ikunc(T)
print(q)
print(robot.fkine(q))

# URDF model
robot = rtb.models.URDF.Panda()
print(robot)
env = rtb.backend.Swift()
env.launch()
env.add(robot)
for qk in qt.q:
    robot.q = qk
    env.step()
