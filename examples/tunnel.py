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

problems = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3]])

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [-0.3728,  1.0259, -0.9876, -1.2082,  2.0677, -0.563, -1.6563]
q1 = [0.2993,  0.1747, -1.7835, -0.8593,  3.1042, -1.2145, -1.3626]
q2 = [0.3537,  0.2079, -1.7168, -0.2937,  3.0224, -0.7083, -1.3286]
q3 = [0.3398, -0.2349, -0.0415, -1.5042,  2.7647, -1.7995,  3.0539]
qs = [q0, q1, q2, q3]

s1 = rp.Shape.Box(
    scale=[0.6, 0.04, 0.6],
    base=sm.SE3(0.65, -0.15, 0.43))

s2 = rp.Shape.Box(
    scale=[0.6, 0.04, 0.6],
    base=sm.SE3(0.65, -0.15, 1.22))

s3 = rp.Shape.Box(
    scale=[0.3, 0.04, 0.2],
    base=sm.SE3(0.8, -0.15, 0.83))

s0 = rp.Shape.Sphere(
    radius=0.05
)

env = rp.backend.Swift()
env.launch()

r = rp.models.PR2()
r.q = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.16825, 0.0, 0.0, 0.0, -0.37279882212870064, 1.0259015008778194,
    -0.9875997438281771, -1.208229229103619, 2.0676739431952065,
    -0.5630237954661839, -1.6563473012595384, 2.248201624865942e-15, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.1102230246251565e-16, 1.1473791555597268,
    -0.2578419004077155, 0.5298918609954418, -2.121201719392923,
    2.198118788614387, -1.4189668927954484, 2.1828521334438378,
    -1.2961853812498703e-14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

R = sm.base.q2r([0.9999523740218402, 0.0, 0.0, 0.00975959466810749])
T = np.eye(4)
T[:3, :3] = R
T = sm.SE3(T)
r.base = sm.SE3(-0.013043247163295746, -0.2435353398323059, 0.0) * T


env.add(r)
# env.add(r, show_robot=False, show_collision=True)
env.add(s0)
env.add(s1)
env.add(s2)
env.add(s3)
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

    if res is None:
        return False

    if res.p[0] < (ob.base.t[0] - (ob.scale[0] / 2.0)) or \
            res.p[0] > (ob.base.t[0] + (ob.scale[0] / 2.0)):
        return False

    # if res.p[1] < (ob.base.t[1] - (ob.scale[1] / 2.0)) or \
    #         res.p[1] > (ob.base.t[1] + (ob.scale[1] / 2.0)):
    #     return False

    if t0[2] < ob.base.t[2]:
        return False

    # print()
    # print(res.p)
    # print(t1)

    return True
    #     return True
    # else:
    #     return False


def shape(T, ob):
    # if not isinstance(ob, rp.Shape.Box):
    #     return False

    ret = True

    # request = fcl.DistanceRequest()
    # result = fcl.DistanceResult()
    # ret = fcl.distance(link.collision[0].co, ob.co, request, result)

    # wTlp = link.collision[0].base * sm.SE3(result.nearest_points[0])
    # wTcp = ob.base * sm.SE3(result.nearest_points[1])
    # lpTcp = wTlp.inv() * wTcp

    # if ret != np.linalg.norm(lpTcp.t):
    #     wTcp = sm.SE3(wTlp.t) * sm.SE3(0, 0, -ret)
    #     lpTcp = wTlp.inv() * wTcp

    # if not wTlp.t[2] > wTcp.t[2]:
    #     return False
    
    # print()
    # print(wTlp.t)
    # print(wTcp.t)
    # if not wTlp.t[0] > wTcp.t[0]:
    #     return False

    if not T.t[2] > ob.base.t[2]:
        return False

    if not T.t[0] < ob.base.t[0]:
        return False



    return ret
#     l = np.argmin(ob.scale)





def link_calc(link, col, ob, q, norm):
    di = 0.16
    ds = 0.05

    request = fcl.DistanceRequest()
    result = fcl.DistanceResult()
    ret = fcl.distance(col.co, ob.co, request, result)

    wTlp = col.base * sm.SE3(result.nearest_points[0])
    wTcp = ob.base * sm.SE3(result.nearest_points[1])
    lpTcp = wTlp.inv() * wTcp

    if ret != np.linalg.norm(lpTcp.t):
        wTcp = sm.SE3(wTlp.t) * sm.SE3(ret * norm)
        # wTcp = sm.SE3(wTlp.t) * sm.SE3(0, 0, -ret)
        lpTcp = wTlp.inv() * wTcp

    d = ret

    if d < di:
        # print()
        # print(d)
        # print(np.linalg.norm(lpTcp.t))
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

    return l_Ain, l_bin, ret


def servo(q0, q1, it):
    r.q[i0:i1] = q1
    r.fkine_all()
    tep = l2._fk.t

    r.q[i0:i1] = q0
    r.qd = np.zeros(r.n)
    env.step(1)

    Tep = r.fkine_graph(q1, l0, l1)

    arrived = False
    i = 0
    Q = 0.1 * np.eye(n + 6)

    # s0.base = sm.SE3(l1._fk.t) * sm.SE3.Tx(0.25)  #r.fkine_graph(r.q[:i1], to_link=l1)
    s0.base = sm.SE3(tep)
    # s0.v = [-0.1, 0, 0, 0, 0, 0]


    while not arrived and i < it:
        q = r.q[i0:i1]
        v, arrived = rp.p_servo(r.fkine_graph(q, l0, l1), Tep, 0.5)
        # v = np.array([-0.1, 0, 0, 0, 0, 0])
        # v[2] = 0

        eTep = r.fkine_graph(q, l0, l1).inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

        Q[n:, n:] = 200 * (1 / e) * np.eye(6)
        # [1/100, 1/100, 1/100, 1/100, 1/100, 1/100] * (100 * np.eye(6))
        Aeq = np.c_[r.jacobe(q, l0, l1), np.eye(6)]
        beq = v.reshape((6,))
        Jm = r.jacobm(q, from_link=l0, to_link=l1).reshape(7,)
        c = np.r_[-Jm, np.zeros(6)]
        # c = np.zeros(13)

        Ain = None
        bin = None

        closest = 1000000
        j = 0
        # for link in links:
        #     if link.jtype == link.VARIABLE:
        #         j += 1

        #     for col in link.collision:
        #         l_Ain, l_bin, ret = link_calc(link, col, s2, q[:j], np.array([0, 0, -1]))
        #         if ret < closest:
        #             closest = ret

        #         if l_Ain is not None and l_bin is not None:
        #             if Ain is None:
        #                 Ain = l_Ain
        #             else:
        #                 Ain = np.r_[Ain, l_Ain]

        #             if bin is None:
        #                 bin = np.array(l_bin)
        #             else:
        #                 bin = np.r_[bin, l_bin]

        j = 0
        # for link in links:
        #     if link.jtype == link.VARIABLE:
        #         j += 1

        #     for col in link.collision:
        #         l_Ain, l_bin, ret = link_calc(link, col, s1, q[:j], np.array([0, -1, 0]))
        #         if ret < closest:
        #             closest = ret

        #         if l_Ain is not None and l_bin is not None:
        #             if Ain is None:
        #                 Ain = l_Ain
        #             else:
        #                 Ain = np.r_[Ain, l_Ain]

        #             if bin is None:
        #                 bin = np.array(l_bin)
        #             else:
        #                 bin = np.r_[bin, l_bin]

        # if closest < 0.2 and shape(l2._fk, s3):
        if plane_int(l2._fk.t, tep, s3):
            v = np.array([-0.1, 0, 0, 0, 0, 0])
            beq = v.reshape((6,))
            Aeq = np.c_[r.jacob0(q, l0, l1), np.eye(6)]

        # plane_int(l2._fk.t, tep, s3)

        # j = 0
        # for link in links:
        #     if link.jtype == link.VARIABLE:
        #         j += 1

        #     for col in link.collision:
        #         l_Ain, l_bin, ret = link_calc(link, col, s0, q[:j], np.array([0, -1, 0]))
        #         if ret < closest:
        #             closest = ret

        #         if l_Ain is not None and l_bin is not None:
        #             if Ain is None:
        #                 Ain = l_Ain
        #             else:
        #                 Ain = np.r_[Ain, l_Ain]

        #             if bin is None:
        #                 bin = np.array(l_bin)
        #             else:
        #                 bin = np.r_[bin, l_bin]

        # print(ret)
        # if ret < 0.15:
        #     beq[0] += -0.1
        # print(np.round(beq, 2))
        # print(closest)


        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        r.qd[i0:i1] = qd[:n]

        # print()
        # print(np.round(v, 2))
        # print(np.round(qd[7:], 2))

        i += 1
        env.step(50)

    return arrived


it_max = 20000

probs = 6
j = 0

for i in range(probs):
    print(problems[i, 0], problems[i, 1])
    ret = servo(qs[problems[i, 0]], qs[problems[i, 1]], 300)

    print(ret)
    if ret:
        j += 1

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
