#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp
import fcl


def link_closest_point(links, ob):

    c_ret = np.inf
    c_res = None
    c_base = None

    for link in links:
        for col in link.collision:
            # col = link.collision[0]
            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            ret = fcl.distance(col.co, sphere.co, request, result)
            if ret < c_ret:
                c_ret = ret
                c_res = result
                c_base = col.base

    # Take the link transform represented in the world frame
    # Multiply it by the translation of the link frame to the nearest point
    # The result is the closest point pose represented in the world frame
    # where the closest point is represented with the links rotational frame
    # rather than the links collision objects rotational frame
    c_wTcp = link._fk * sm.SE3((c_base * sm.SE3(result.nearest_points[0])).t)

    return c_ret, c_res, c_wTcp


# def link_closest_point(link, ob):

#     c_ret = np.inf
#     c_res = None
#     c_base = None
#     c_type = None

#     for col in link.collision:
#         # col = link.collision[0]
#         request = fcl.DistanceRequest()
#         result = fcl.DistanceResult()
#         ret = fcl.distance(col.co, sphere.co, request, result)
#         if ret < c_ret:
#             c_ret = ret
#             c_res = result
#             c_base = col.base
#             c_type = col.stype

#     # if c_type == 'cylinder':
#     # print(result.nearest_points[0])

#     # result.nearest_points[0][2] = 0

#     # Take the link transform represented in the world frame
#     # Multiply it by the translation of the link frame to the nearest point
#     # The result is the closest point pose represented in the world frame
#     # where the closest point is represented with the links rotational frame
#     # rather than the links collision objects rotational frame
#     c_wTcp = link._fk * sm.SE3((c_base * sm.SE3(result.nearest_points[0])).t)

#     return c_ret, c_res, c_wTcp


def closer(ret1, res1, ret2, res2):
    if ret1 < ret2:
        return ret1, res1
    else:
        return ret2, res2


env = rp.backend.Swift()
env.launch()

panda = rp.models.Panda()
panda.q = panda.qr
# panda.q[4] += 0.1
# Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)
Tep = panda.fkine() * sm.SE3.Tz(0.6) * sm.SE3.Tx(-0.1)  # * sm.SE3.Ty(-0.1)

sphere = rp.Shape.Sphere(0.05, sm.SE3(0.5, 0, 0.2))
sphere.wT = sm.SE3()

arrived = False
env.add(panda, show_collision=True, show_robot=False)
env.add(sphere)
time.sleep(1)

dt = 0.05
ps = 0.05
pi = 0.6

# env.record_start('file.webm')

while not arrived:

    start = time.time()
    v, arrived = rp.p_servo(panda.fkine(), Tep, 1.0)
    # panda.qd = np.linalg.pinv(panda.jacobe()) @ v
    # env.step(5)

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

    Q = np.eye(panda.n + 6)
    Q[:panda.n, :panda.n] *= Y
    Q[panda.n:, panda.n:] = (1 / e) * np.eye(6)
    Aeq = np.c_[panda.jacobe(), np.eye(6)]
    beq = v.reshape((6,))
    c = np.r_[-panda.jacobm().reshape((panda.n,)), np.zeros(6)]


    # Get closest link
    linkA = panda._fkpath[-1]
    linkB = panda._fkpath[-2]
    linkC = panda._fkpath[-3]
    panda.fkine_all()
    retA, resA, wTcp = link_closest_point([linkA, linkB], sphere)
    cpTc = wTcp.inv() * (sphere.base * sm.SE3(resA.nearest_points[1]))


    d0 = np.linalg.norm(cpTc.t)
    n0 = cpTc.t / d0

    # Distance Jacobian
    ds = 0.05
    di = 0.6

    if d0 <= di:
        nh0 = np.expand_dims(np.r_[n0, 0, 0, 0], axis=0)
        Ain = np.c_[nh0 @ panda.jacobe(), np.zeros((1, 6))]
        bin = np.array([0.1 * (d0 - ds) / (di - ds)])
    else:
        Ain = None
        bin = None

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

    if np.any(np.isnan(qd)):
        panda.qd = panda.qz
    else:
        panda.qd = qd[:panda.n]

    env.step(25)

    stop = time.time()
    if stop - start < dt:
        time.sleep(dt - (stop - start))

# env.record_stop()

# Uncomment to stop the plot from closing
# env.hold()
