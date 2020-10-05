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

q0 = [-0.2762, 0.894, -0.4462, -1.6105,  2.328, -1.0881, -2.3999]
q1 = [-0.571, -0.3534, -1.7684, -1.5384, -2.7693, -1.5537, -1.4818]
q2 = [-0.1708,  0.093, -1.1193, -1.0388,  2.8681, -1.4571, -1.9924]
q3 = [0.2855, -0.198, -3.2144, -1.0791, -2.0565, -1.1219, -0.6414]
q4 = [-0.5471,  0.1817, -1.835, -1.6187,  3.0026, -1.7675, -1.3979]
q5 = [0.1713,  0.377, -0.5826, -0.5771, -1.5264, -0.4143,  2.0764]
q6 = [0.514,  0.2662, -1.2524, -0.6177,  2.9156, -0.2591, -1.7356]
q7 = [0.5512, -0.3535, -1.2124, -0.4724, -2.1021, -0.5965, -2.8023]
q8 = [0.5272,  0.7193, -0.9876, -0.5453,  1.2938, -0.3151, -0.5195]
q9 = [-0.3728,  1.0259, -0.9876, -1.2082,  2.0042, -1.3781, -1.6173]
qs = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]

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


s0 = rp.Shape.Sphere(
    radius=0.05
)

env = rp.backend.Swift()
env.launch()

r = rp.models.PR2()
r.q = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16825, 0.0,
    0.0, 0.0, -0.2761549481623158, 0.8939542514393661, -0.4462472669817504,
    -1.6105027440487039, 2.328001213611822, -1.0881333252440992,
    -2.3998853474978716, 2.248201624865942e-15, 0.0, 0.0, 0.0, 0.0, 0.0,
    -1.1102230246251565e-16, 1.1473791555597268, -0.2578419004077155,
    0.5298918609954418, -2.121201719392923, 2.198118788614387,
    -1.4189668927954484, 2.1828521334438378, -1.174060848541103e-14, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.6653345369377348e-16, 0.0]

R = sm.base.q2r([0.9074254167930225, 0.0, 0.0, -0.42021317561210453])
T = np.eye(4)
T[:3, :3] = R
T = sm.SE3(T)
r.base = sm.SE3(-0.0048143863677978516, 0.039366304874420166, 0.0) * T

env.add(r)
# env.add(r, show_robot=False, show_collision=True)
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
        # j = 0
        # for link in links:
        #     if link.jtype == link.VARIABLE:
        #         j += 1

        #     for col in link.collision:
        #         l_Ain, l_bin, ret = link_calc(link, col, s3, q[:j], np.array([0, 0, -1]))
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

        # j = 0
        # for link in links:
        #     if link.jtype == link.VARIABLE:
        #         j += 1

        #     for col in link.collision:
        #         l_Ain, l_bin, ret = link_calc(link, col, s6, q[:j], np.array([0, -1, 0]))
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

        # # if closest < 0.2 and shape(l2._fk, s3):
        # if plane_int(l2._fk.t, tep, s3):
        #     v = np.array([-0.1, 0, 0, 0, 0, 0])
        #     beq = v.reshape((6,))
        #     Aeq = np.c_[r.jacob0(q, l0, l1), np.eye(6)]

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

probs = 45
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
