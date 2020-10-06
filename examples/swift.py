#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp
import pybullet as p

ta = 0
tb = 0
count = 0

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])

env = rp.backend.Swift()
env.launch()

panda = rp.models.Panda()
panda.q = panda.qr
# panda.q[4] += 0.1
# Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)
# Tep = panda.fkine() * sm.SE3.Tz(0.6) * sm.SE3.Tx(-0.1)  # * sm.SE3.Ty(-0.1)
Tep = panda.fkine() * sm.SE3.Tz(0.45) * sm.SE3.Tx(0.1)


# s1 = rp.Shape.Sphere(0.05, sm.SE3(0.5, 0, 0.2))
s1 = rp.Shape.Sphere(0.02, Tep)
s2 = rp.Shape.Sphere(0.05, sm.SE3(0.10, 0.25, 0.6))
s3 = rp.Shape.Sphere(0.05, sm.SE3(0.25, -0.25, 0.1))
# s1.v = [0.0, 0.0, 0.08, 0, 0, 0]
s2.v = [0, -0.25, 0, 0, 0, 0]
# s3.v = [0.2, 0.2, 0, 0, 0, 0]

s = [s2]

arrived = False
# env.add(panda, show_collision=True, show_robot=False)
env.add(panda)
env.add(s1)
env.add(s2)
# env.add(s3)
# env.add(s4)
time.sleep(1)

dt = 0.05
ps = 0.05
pi = 0.6

lb = -qdmax
ub = qdmax

links, n = panda.get_path(panda.base_link, panda.ee_link)


def link_calc(link, col, ob, q):
    di = 0.3
    ds = 0.05

    ret = p.getClosestPoints(col.co, ob.co, di)

    if len(ret) > 0:
        ret = ret[0]
        wTlp = sm.SE3(ret[5])
        wTcp = sm.SE3(ret[6])
        lpTcp = wTlp.inv() * wTcp

        d = ret[8]

        n = lpTcp.t / d
        nh = np.expand_dims(np.r_[n, 0, 0, 0], axis=0)

        Je = panda.jacobe(q, to_link=link, offset=col.base)
        n = Je.shape[1]
        dp = nh @ ob.v
        l_Ain = np.zeros((1, 13))
        l_Ain[0, :n] = nh @ Je
        l_bin = (5 * (d - ds) / (di - ds)) + dp
    else:
        l_Ain = None
        l_bin = None
        d = 1000
        wTcp = None

    return l_Ain, l_bin, d, wTcp


# def link_closest_point(links, ob):

#     c_ret = np.inf
#     c_res = None
#     c_base = None

#     for link in links:
#         for col in link.collision:
#             # col = link.collision[1]
#             request = fcl.DistanceRequest()
#             result = fcl.DistanceResult()
#             ret = fcl.distance(col.co, sphere.co, request, result)
#             if ret < c_ret:
#                 c_ret = ret
#                 c_res = result
#                 c_base = col.base

#     # Take the link transform represented in the world frame
#     # Multiply it by the translation of the link frame to the nearest point
#     # The result is the closest point pose represented in the world frame
#     # where the closest point is represented with the links rotational frame
#     # rather than the links collision objects rotational frame
#     wTlp = link._fk * sm.SE3((c_base * sm.SE3(c_res.nearest_points[0])).t)
#     # print(wTlp)
#     # wTlp = sm.SE3() * sm.SE3(wTlp.t)
#     # print(wTlp)

#     # wTlp = link._fk * sm.SE3(c_base.t)

#     # wTcp = sm.SE3() * sm.SE3((ob.base * sm.SE3(c_res.nearest_points[1])).t)
#     wTcp = ob.base * sm.SE3(c_res.nearest_points[1])

#     lpTcp = wTlp.inv() * wTcp

#     return c_ret, c_res, lpTcp

# env.record_start('file.webm')
while not arrived:
    q = panda.q[:7]
    v, arrived = rp.p_servo(panda.fkine(), Tep, 1)
    eTep = panda.fkine().inv() * Tep
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))
    Y = 0.01

    # Ain = np.zeros((panda.n + 6, panda.n + 6))
    # bin = np.zeros(panda.n + 6)

    # for i in range(panda.n):
    #     if panda.q[i] - panda.qlim[0, i] <= pi:
    #         bin[i] = -1.0 * (((panda.qlim[0, i] - panda.q[i]) + ps) / (pi - ps))
    #         Ain[i, i] = -1
    #     if panda.qlim[1, i] - panda.q[i] <= pi:
    #         bin[i] = ((panda.qlim[1, i] - panda.q[i]) - ps) / (pi - ps)
    #         Ain[i, i] = 1
    # print()
    # print(np.round(v, 2))
    n = 7
    Q = np.eye(n + 6)
    Q[:n, :n] *= Y
    Q[n:, n:] = (1 / e) * np.eye(6)
    # Q[n:, n:] = [1/10, 1/100, 1/100, 1/100, 1/100, 1/100] * np.eye(6)
    Aeq = np.c_[panda.jacobe(), np.eye(6)]
    beq = v.reshape((6,))
    Jm = panda.jacobm().reshape((7,))
    c = np.r_[-Jm[:7], np.zeros(6)]
    # c = np.zeros(13)

    # l_groups = []
    # j = -1
    # for link in panda._fkpath:

    #     if link.jtype:
    #         j += 1
    #         l_groups.append([])

    #     if j >= 0:
    #         l_groups[j].append(link)

    # Distance Jacobian
    Ain = None
    bin = None
    j = 0    
    for link in links:
        if link.jtype == link.VARIABLE:
            j += 1

        for col in link.collision:
            for obj in s:
                l_Ain, l_bin, ret, _ = link_calc(link, col, obj, q[:j])

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.r_[Ain, l_Ain]

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.r_[bin, l_bin]

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    # qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

    # print(np.round(qd[7:], 2))

    if np.any(np.isnan(qd)):
        panda.qd = panda.qz
    else:
        panda.qd = qd[:7]

    env.step(15)

# print(ta/count)
# print(tb/count)
