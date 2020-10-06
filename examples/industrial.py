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

problems = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 9],
    [0, 10],
    [0, 11],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [1, 10],
    [1, 11],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [2, 7],
    [2, 8],
    [2, 9],
    [2, 10],
    [2, 11],
    [3, 4],
    [3, 5],
    [3, 6],
    [3, 7],
    [3, 8],
    [3, 9],
    [3, 10],
    [3, 11],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [4, 9],
    [4, 10],
    [4, 11],
    [5, 6],
    [5, 7],
    [5, 8],
    [5, 9],
    [5, 10],
    [5, 11],
    [6, 7],
    [6, 8],
    [6, 9],
    [6, 10],
    [6, 11],
    [7, 8],
    [7, 9],
    [7, 10],
    [7, 11],
    [8, 9],
    [8, 10],
    [8, 11],
    [9, 10],
    [9, 11],
    [10, 11]])

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [0.4522, -0.0688,  0.4516, -1.3564,  2.6366, -1.2885,  2.7515]
q1 = [0.5645, -0.2949, -0.8384, -0.4594,  2.9742, -1.1313, -2.3144]
q2 = [0.2846, -0.3534, -2.2822, -1.2794, -2.6088, -1.298, -1.1206]
q3 = [0.281,  0.7666,  0.0846, -1.8104,  2.8603, -1.1027, -3.1227]
q4 = [0.0514, -0.3534, -2.8542, -1.7384, -2.95, -1.3661, -0.3652]
q5 = [0.2874, -0.0535, -2.3339, -1.2294, -2.8416, -1.3918, -0.9476]
q6 = [0.5259,  0.9754, -0.2409, -0.9607,  1.7601, -0.748, -1.6548]
q7 = [-0.4355,  1.2821, -0.5184, -1.4671,  2.7845, -0.2334, -2.709]
q8 = [-1.0536, -0.3436, -1.6586, -1.4779, -2.7631, -0.4247, -1.6377]
q9 = [-0.7221, -0.3534, -1.639, -1.7309, -2.8092, -0.9864, -1.5293]
q10 = [-0.3357, -0.1715, -0.3289, -2.0001,  1.6523, -1.9265,  2.5474]
q11 = [-0.3888,  1.0477, -1.1405, -0.7096,  0.9253, -0.5049, -0.3575]
qs = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11]

s1 = rp.Shape.Box(
    scale=[0.4, 1.4, 0.02],
    base=sm.SE3(0.75, -0.1, 0.7))

s2 = rp.Shape.Box(
    scale=[1.1, 0.3, 0.02],
    base=sm.SE3(0.4, -0.65, 0.7))

s3 = rp.Shape.Box(
    scale=[0.02, 0.2, 0.3],
    base=sm.SE3(0.1, -0.7, 0.85))

s4 = rp.Shape.Box(
    scale=[0.02, 0.2, 0.3],
    base=sm.SE3(0.3, -0.7, 0.85))

s5 = rp.Shape.Box(
    scale=[0.02, 0.2, 0.3],
    base=sm.SE3(0.5, -0.7, 0.85))

s6 = rp.Shape.Box(
    scale=[0.42, 0.3, 0.02],
    base=sm.SE3(0.3, -0.65, 0.71))

s7 = rp.Shape.Box(
    scale=[0.02, 0.1, 0.06],
    base=sm.SE3(0.1, -0.55, 0.73))

s8 = rp.Shape.Box(
    scale=[0.02, 0.1, 0.06],
    base=sm.SE3(0.3, -0.55, 0.73))

s9 = rp.Shape.Box(
    scale=[0.02, 0.1, 0.06],
    base=sm.SE3(0.5, -0.55, 0.73))

s10 = rp.Shape.Box(
    scale=[0.42, 0.02, 0.06],
    base=sm.SE3(0.3, -0.5, 0.73))

s11 = rp.Shape.Box(
    scale=[0.42, 0.02, 0.06],
    base=sm.SE3(0.3, -0.59, 0.93))

s12 = rp.Shape.Box(
    scale=[0.42, 0.02, 0.34],
    base=sm.SE3(0.3, -0.79, 0.87))

s13 = rp.Shape.Box(
    scale=[0.42, 0.2, 0.02],
    base=sm.SE3(0.3, -0.7, 0.9))

s14 = rp.Shape.Box(
    scale=[0.3, 0.02, 0.45],
    base=sm.SE3(0.8, 0.25, 0.925))

s15 = rp.Shape.Box(
    scale=[0.3, 0.02, 0.45],
    base=sm.SE3(0.8, -0.25, 0.925))

s16 = rp.Shape.Box(
    scale=[0.3, 0.52, 0.02],
    base=sm.SE3(0.8, -0.00, 1.16))

s17 = rp.Shape.Box(
    scale=[0.42, 0.02, 0.34],
    base=sm.SE3(0.3, -0.79, 0.87))

s = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12,
     s13, s14, s15, s16, s17]

s0 = rp.Shape.Sphere(
    radius=0.05
)

s00 = rp.Shape.Sphere(
    radius=0.05
)

se = rp.Shape.Sphere(
    radius=0.02,
    base=sm.SE3(0.18, 0.01, 0)
)

env = rp.backend.Swift()
env.launch()

r = rp.models.PR2()
r.q = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.16825, 0.0, 0.0, 0.0, 0.4521781848795534, -0.06875607890205337,
    0.45157478971034287, -1.356432823197765, 2.63664188822003,
    -1.2884530258626397, 2.7514905004419816, 2.248201624865942e-15, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.1102230246251565e-16, 1.1473791555597268,
    -0.2578419004077155, 0.5298918609954418, -2.121201719392923,
    2.198118788614387, -1.4189668927954484, 2.1828521334438378,
    -4.08006961549745e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 7.549516567451064e-15,
    0.0]

R = sm.base.q2r([0.9999554696772218, 0.0, 0.0, 0.009437089731840358])
T = np.eye(4)
T[:3, :3] = R
T = sm.SE3(T)
# r.base = sm.SE3(0.10682493448257446, -0.09225612878799438, 0.0) * T
r.base = sm.SE3(0, -0.09225612878799438, 0.0) * T
# r.base = sm.SE3(0, 0, 0.0) * T

env.add(r)
# env.add(r, show_robot=False, show_collision=True)
env.add(se)
env.add(s00)
env.add(s0)
env.add(s1)
env.add(s2)
env.add(s3)
env.add(s4)
env.add(s5)
env.add(s6)
env.add(s7)
env.add(s8)
env.add(s9)
env.add(s10)
env.add(s11)
env.add(s12)
env.add(s13)
env.add(s14)
env.add(s15)
env.add(s16)
env.add(s17)
time.sleep(1)

# ETS number 43
l0 = r.elinks['r_shoulder_pan_joint']

# ETS number 51
l1 = r.elinks['r_wrist_roll_joint']
l2 = r.elinks['r_gripper_joint']

i0 = 16
i1 = i0 + 7
links, n = r.get_path(l0, l1)


def plane_int(t0, t1, ob):

    point = ob.base.t
    normal = np.zeros(3)
    normal[np.argmin(ob.scale)] = 1
    plane = sm.geom3d.Plane.PN(point, normal)

    line = sm.Plucker.PQ(t0, t1)

    res = line.intersect_plane(plane)

    off = 0.1

    if normal[2]:
        if res is None:
            return False

        if res.p[0] < (ob.base.t[0] - (ob.scale[0] / 2.0) - off) or \
                res.p[0] > (ob.base.t[0] + (ob.scale[0] / 2.0) + off):
            return False

        if res.p[1] < (ob.base.t[1] - (ob.scale[1] / 2.0) - off) or \
                res.p[1] > (ob.base.t[1] + (ob.scale[1] / 2.0) + off):
            return False

        above0 = t0[2] > ob.base.t[2]
        above1 = t1[2] > ob.base.t[2]

        if above0 and above1 or (not above0 and not above1):
            return False

        return True
    elif normal[1]:
        if res is None:
            return False

        if res.p[0] < (ob.base.t[0] - (ob.scale[0] / 2.0) - off) or \
                res.p[0] > (ob.base.t[0] + (ob.scale[0] / 2.0) + off):
            return False

        if res.p[2] < (ob.base.t[2] - (ob.scale[2] / 2.0) - off) or \
                res.p[2] > (ob.base.t[2] + (ob.scale[2] / 2.0) + off):
            return False

        above0 = t0[1] > ob.base.t[1]
        above1 = t1[1] > ob.base.t[1]

        if above0 and above1 or (not above0 and not above1):
            return False

        return True
    else:
        return False


def link_calc(link, col, ob, q):
    di = 0.3
    ds = 0.02

    ret = p.getClosestPoints(col.co, ob.co, di)

    if len(ret) > 0:
        ret = ret[0]
        wTlp = sm.SE3(ret[5])
        wTcp = sm.SE3(ret[6])
        lpTcp = wTlp.inv() * wTcp

        d = ret[8]

        n = lpTcp.t / d
        nh = np.expand_dims(np.r_[n, 0, 0, 0], axis=0)

        Je = r.jacobe(q, from_link=l0, to_link=link, offset=col.base)
        n = Je.shape[1]
        dp = nh @ ob.v
        l_Ain = np.zeros((1, 13))
        l_Ain[0, :n] = nh @ Je
        l_bin = (1 * (d - ds) / (di - ds)) + dp
    else:
        l_Ain = None
        l_bin = None
        d = 1000
        wTcp = None

    return l_Ain, l_bin, d, wTcp


def servo(q0, q1, it):
    r.q[i0:i1] = q1
    r.fkine_all()
    tep = l2._fk.t

    r.q[i0:i1] = q0
    r.fkine_all()
    r.qd = np.zeros(r.n)
    env.step(1)

    Tep = r.fkine_graph(q1, l0, l1)

    arrived = False
    i = 0
    Q = 0.1 * np.eye(n + 6)

    s0.base = sm.SE3(tep)

    while not arrived and i < it:
        q = r.q[i0:i1]
        v, arrived = rp.p_servo(r.fkine_graph(q, l0, l1), Tep, 1, 0.25)

        se._wT = l1._fk
        # v = np.array([-0.1, 0, 0, 0, 0, 0])
        # v[2] = 0

        eTep = r.fkine_graph(q, l0, l1).inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

        Q[n:, n:] = (1 / e) * np.eye(6)
        Aeq = np.c_[r.jacobe(q, l0, l1), np.eye(6)]
        beq = v.reshape((6,))
        Jm = r.jacobm(q, from_link=l0, to_link=l1).reshape(7,)
        c = np.r_[-Jm, np.zeros(6)]

        Ain = None
        bin = None

        closest = 1000000
        closest_obj = None
        closest_p = None
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

        for obj in s:
            l_Ain, l_bin, ret, wTcp = link_calc(l1, se, obj, r.q[i0:i1])
            if ret < closest:
                closest = ret
                closest_obj = obj
                closest_p = wTcp

            if l_Ain is not None and l_bin is not None:
                if Ain is None:
                    Ain = l_Ain
                else:
                    Ain = np.r_[Ain, l_Ain]

                if bin is None:
                    bin = np.array(l_bin)
                else:
                    bin = np.r_[bin, l_bin]

        s00.base = closest_p
        if plane_int(se.wT.t, tep, closest_obj):
            # v = np.array([-0.1, 0, 0, 0, 0, 0])
            v[0] = -0.3
            # v[2] /= 10
            beq = v.reshape((6,))
            Aeq = np.c_[r.jacob0(q, l0, l1), np.eye(6)]

        try:
            qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        except (ValueError, TypeError):
            print("Value Error")
            break

        r.qd[i0:i1] = qd[:n]

        i += 1

        env.step(50)

    return arrived


it_max = 20000

probs = 66
j = 0

start = 3

for i in range(start, probs):
    print(problems[i, 0], problems[i, 1], i)
    ret = servo(qs[problems[i, 0]], qs[problems[i, 1]], 500)

    print(ret)
    if ret:
        j += 1
    print(j)

# i = 5
# print(problems[i, 0], problems[i, 1], i)
# ret = servo(qs[problems[i, 0]], qs[problems[i, 1]], 300)
# env.step(1)
# print(j)

# Problems
# 1 -> 4 - 13
