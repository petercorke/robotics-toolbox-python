# this is the example code from the t0p-level README..d
from spatialmath import SE3
import roboticstoolbox as rtb
import swift

robot = rtb.models.DH.Panda()
print(robot)
T = robot.fkine(robot.qz)
print(T)

# IK

T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ikine_LMS(T)  # solve IK, ignore additional outputs
print(sol.q)  # display joint angles
# FK shows that desired end-effector pose was achieved
print(robot.fkine(sol.q))


qtraj = rtb.jtraj(robot.qz, sol.q, 50)
robot.plot(qtraj.q, movie="panda1.gif")

# URDF + Swift version
dt = 0.050  # simulation timestep in seconds
robot = rtb.models.URDF.Panda()
print(robot)

env = swift.Swift()   # instantiate 3D browser-based visualizer
env.launch("chrome")        # activate it
env.add(robot)              # add robot to the 3D scene
env.start_recording("panda2", 1 / dt)
for qk in qtraj.q:             # for each joint configuration on trajectory
    robot.q = qk          # update the robot state
    env.step()            # update visualization
env.stop_recording()

# ffmpeg -i panda2.webm -vf "scale=iw*.5:ih*.5" panda2.gif
