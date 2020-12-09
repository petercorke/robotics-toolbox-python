#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm

# Launch the simulator Swift
# env = rtb.backends.Swift()
# env.launch()

# # Create a Panda robot object
# robot = rtb.models.Puma560()
# env.add(robot)

# env.hold()

# print(panda)
# print(panda.base_link)
# print(panda.ee_links)

# path, n = panda.get_path(panda.base_link, panda.ee_links[0])

# q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
# panda.q = q1

# print(panda.fkine())

# for link in path:
#     print(link.name)

# print(panda.get_path(panda.base_link, panda.ee_links[0])[0])

# print(panda.links[5].A(0))

# # Set joint angles to ready configuration
# panda.q = panda.qr

# Add the Panda to the simulator
# env.add(panda)


# while 1:
#     pass

panda = rtb.models.Panda()
robot = rtb.models.DH.Panda()

panda.q = panda.qr

T = panda.fkine(panda.qr)
# T = sm.SE3(0.8, 0.2, 0.1) * sm.SE3.OA([0, 1, 0], [0, 0, -1])

sol = robot.ikine_LM(T)         # solve IK


qt = rtb.jtraj(robot.qz, sol.q, 50)

env = rtb.backends.Swift()  # instantiate 3D browser-based visualizer
env.launch()                # activate it
env.add(panda)              # add robot to the 3D scene
for qk in qt.q:             # for each joint configuration on trajectory
    panda.q = qk          # update the robot state
    env.step()            # update visualization
