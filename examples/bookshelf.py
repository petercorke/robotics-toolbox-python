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
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [2, 7],
    [2, 8],
    [2, 9],
    [3, 4],
    [3, 5],
    [3, 6],
    [3, 7],
    [3, 8],
    [3, 9],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [4, 9],
    [5, 6],
    [5, 7],
    [5, 8],
    [5, 9],
    [6, 7],
    [6, 8],
    [6, 9],
    [7, 8],
    [7, 9],
    [8, 9]])

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [-0.5653, -0.1941, -1.2602, -0.7896, -2.3227, -0.3919, -2.5173]
q1 = [-0.1361, -0.1915, -1.2602, -0.8652, -2.8852, -0.7962, -2.039]
q2 = [0.2341, -0.2138, -1.2602, -0.4709, -3.0149, -0.7505, -2.0164]
q3 = [0.1584,  0.3429, -1.2382, -0.9829, -2.0892, -1.6126, -0.5582]
q4 = [0.3927,  0.1763, -1.2382, -0.1849, -1.96, -1.4092, -1.0492]
q5 = [-0.632,  0.5012, -1.2382, -0.8353, 2.2571, -0.1041,  0.3066]
q6 = [0.1683,  0.7154, -0.4195, -1.0496, 2.4832, -0.6028, -0.6401]
q7 = [-0.1198,  0.5299, -0.6291, -0.4348, 2.1715, -1.6403,  1.8299]
q8 = [0.2743,  0.4088, -0.5291, -0.4304, 2.119, -1.9994, 1.7162]
q9 = [0.2743,  0.4088, -0.5291, -0.4304, -0.9985, -1.0032, -1.7278]
qs = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]

s1 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 0.20))

s2 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 0.60))

s3 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 1.00))

s4 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 1.40))

s5 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(0.95, 0.55, 0.7))

s6 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(0.95, -0.55, 0.7))

s0 = rp.Shape.Sphere(
    radius=0.05
)

s = [s1, s2, s3, s4, s5, s6]

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
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.16825, 0.0, 0.0, 0.0, -0.5652894131595758, -0.1940789551546196,
    -1.260201738335192, -0.7895653603354864, -2.322747882942366,
    -0.3918504494615993, -2.5173485998351066, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 2.1347678577910827, 0.05595277251194286, 0.48032314980402596,
    -2.0802263633096487, 1.2294916701952125, -0.8773017824611689,
    2.932954218704465, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# r.base = sm.SE3(0.11382412910461426, 0.0, 0.0)

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
        l_bin = (5 * (d - ds) / (di - ds)) + dp
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

        eTep = r.fkine_graph(q, l0, l1).inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

        Q[n:, n:] = 10 * (1 / e) * np.eye(6)
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
            v[0] = -0.4
            v[3:] /= 10
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

probs = 45
j = 20

for i in range(j, probs):
    print(problems[i, 0], problems[i, 1])
    ret = servo(qs[problems[i, 0]], qs[problems[i, 1]], 500)

    print(ret)
    if ret:
        j += 1
        print(j)

print(j)


# for i in range(10):
#     # for j in range(i+1, 10):
#     #     print(i, j)
#     #     print(servo(qs[i], qs[j], 100))
#     print(servo(qs[i], qs[9], it_max))

# 1 -> 9
# 2 -> 9
# 5 -> 9
# 6 -> 9
