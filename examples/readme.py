# this is the example code from the t0p-level README..d
import roboticstoolbox as rtb

robot = rtb.models.DH.Panda()
print(robot)
T = robot.fkine(robot.qz)
print(T)

# IK
from spatialmath import SE3

T = SE3(0.8, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
q_pickup, *_ = robot.ikine(T)  # solve IK, ignore additional outputs
print(q_pickup)# display joint angles
print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved


qt = rtb.trajectory.jtraj(robot.qz, q_pickup, 50)
robot.plot(qt.q, movie="panda1.gif")

# URDF + Swift version
dt = 0.050  # simulation timestep in seconds
robot = rtb.models.URDF.Panda()
print(robot)

env = rtb.backends.Swift()   # instantiate 3D browser-based visualizer
env.launch("chrome")        # activate it
env.add(robot)              # add robot to the 3D scene
env.start_recording("panda2", 1 / dt)
for qk in qt.q:             # for each joint configuration on trajectory
      robot.q = qk          # update the robot state
      env.step()            # update visualization
env.stop_recording()

# ffmpeg -i panda2.webm -vf "scale=iw*.5:ih*.5" panda2.gif
