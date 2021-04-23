#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)


r = rtb.models.YuMi()

# r.q = [
#     -0.6, -0.3, -0.3, 0.2, 0, 0.3,
#     0, 0,
#     0.6, -0.3, 0.3, 0.2, 0, 0.3,
#     0, 0
# ]

env.add(r)


ev_r = [0, 0.1, 0, 0, 0, 0]
ev_l = [0, -0.001, 0, 0, 0, 0]

for i, link in enumerate(r.links):
    print(link)
    print(i)


# print()

# path, n, _ = r.get_path(end=r.grippers[0])


# def path_to_q(q_partial, robot, path):
#     j = 0
#     for link in path:
#         if link.isjoint:
#             q_whole[link.jindex] = q_partial[j]
#             j += 1


# for link in path:
#     print(link.jindex)


# for i in range(10000):
#     qd_r = np.linalg.pinv(r.jacob0(r.q, end=r.grippers[0])) @ ev_r
# #     # qd_l = np.linalg.pinv(r.jacob0(r.q, end=r.grippers[1])) @ ev_l

#     r.qd[:6] = qd_r[:6]

#     # r.qd = resolve_qd(path, qd_whole, qd_subset)

#     print(qd_r)
# #     # print(qd_l)
# #     # r.qd[3] = 1
#     env.step(0.001)

env.hold()

# ur = rtb.models.UR5()
# ur.base = sm.SE3(0.3, 1, 0) * sm.SE3.Rz(np.pi / 2)
# ur.q = [-0.4, -np.pi / 2 - 0.3, np.pi / 2 + 0.3, -np.pi / 2, -np.pi / 2, 0]
# env.add(ur)

# # lbr = rtb.models.LBR()
# # lbr.base = sm.SE3(1.8, 1, 0) * sm.SE3.Rz(np.pi / 2)
# # lbr.q = lbr.qr
# # env.add(lbr)

# # k = rtb.models.KinovaGen3()
# # k.q = k.qr
# # k.q[0] = np.pi + 0.15
# # k.base = sm.SE3(0.7, 1, 0) * sm.SE3.Rz(np.pi / 2)
# # env.add(k)

# # panda = rtb.models.Panda()
# # panda.q = panda.qr
# # panda.base = sm.SE3(1.2, 1, 0) * sm.SE3.Rz(np.pi / 2)
# # env.add(panda, show_robot=True)

# r = rtb.models.YuMi()
# env.add(r)


# env.hold()

# import roboticstoolbox as rtb
# from spatialmath import SE3
# from swift import Swift

# total_robots = 1000
# robots = []  # list of robots
# d = 1  # robot spacing
# env = Swift(_dev=True)
# env.launch()
# for i in range(total_robots):
#     base = SE3(d * (i % 2), d * (i // 2), 0.0)  # place them on grid
#     robot = rtb.models.URDF.Puma560()
#     robot.base = base
#     robots.append(robot)
#     env.add(robots[i])

# env.hold()
