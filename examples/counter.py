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
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [2, 7],
    [2, 8],
    [3, 4],
    [3, 5],
    [3, 6],
    [3, 7],
    [3, 8],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [5, 6],
    [5, 7],
    [5, 8],
    [6, 7],
    [6, 8],
    [7, 8]])

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [-0.8452,  0.0813, -1.8312, -1.555,  2.6911, -0.8326,  2.068]
q1 = [-0.4134, -0.238, -3.6504, -1.1768,  2.7225, -1.2706, -2.3709]
q2 = [-0.9547,  0.3356, -2.3151, -0.9126,  1.8166, -0.8724, -3.1287]
q3 = [-0.496, -0.2946, -2.626, -1.5671, -0.9644, -0.5307,  0.5828]
q4 = [-0.978, -0.235, -1.3629, -1.282, -2.2903, -0.4913,  0.9081]
q5 = [-0.3043, -0.1995,  0.4997, -0.9161, -3.0128, -1.2772, -0.4844]
q6 = [-0.0826, -0.3115, -0.7685, -1.0468, -2.8332, -1.2915,  0.7087]
q7 = [-0.9493, -0.2259, -1.2924, -1.2902, -2.2911, -0.5655, -2.1449]
q8 = [-0.0077, -0.1813, -1.2825, -0.2072, -2.475, -0.3674, -2.5659]
qs = [q0, q1, q2, q3, q4, q5, q6, q7, q8]

s1 = rp.Shape.Box(
    scale=[0.60, 0.71, 0.02],
    base=sm.SE3(0.30, 0.355, 1.50))

s2 = rp.Shape.Box(
    scale=[0.70, 0.02, 0.90],
    base=sm.SE3(0.65, 0.01, 1.06))

s3 = rp.Shape.Box(
    scale=[0.59, 0.70, 0.02],
    base=sm.SE3(0.30, 0.35, 1.23))

s4 = rp.Shape.Box(
    scale=[0.59, 0.70, 0.02],
    base=sm.SE3(0.30, 0.35, 0.92))

s5 = rp.Shape.Box(
    scale=[0.59, 0.70, 0.02],
    base=sm.SE3(0.30, 0.35, 0.60))

s6 = rp.Shape.Box(
    scale=[0.02, 0.70, 0.60],
    base=sm.SE3(0.60, 0.35, 0.30))

s7 = rp.Shape.Box(
    scale=[0.07, 0.03, 0.03],
    base=sm.SE3(0.63, 0.20, 0.55))

s8 = rp.Shape.Box(
    scale=[0.07, 0.03, 0.03],
    base=sm.SE3(0.63, 0.50, 0.55))

s9 = rp.Shape.Box(
    scale=[0.03, 0.30, 0.03],
    base=sm.SE3(0.65, 0.35, 0.55))

s10 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.5],
    base=sm.SE3(0.30, 0.01, 0.75))

s11 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.5],
    base=sm.SE3(0.30, 0.70, 0.75))

s12 = rp.Shape.Box(
    scale=[0.02, 0.70, 1.5],
    base=sm.SE3(0.02, 0.35, 0.75))

s13 = rp.Shape.Box(
    scale=[0.60, 0.60, 0.02],
    base=sm.SE3(0.30, 1.01, 0.75))

s14 = rp.Shape.Box(
    scale=[0.60, 0.52, 0.02],
    base=sm.SE3(0.30, 2.14, 0.75))

s15 = rp.Shape.Box(
    scale=[0.60, 0.60, 0.02],
    base=sm.SE3(0.30, 1.60, 0.50))

s16 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.25],
    base=sm.SE3(0.30, 1.30, 0.625))

s17 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.25],
    base=sm.SE3(0.30, 1.90, 0.625))

s18 = rp.Shape.Box(
    scale=[0.02, 2.4, 2.0],
    base=sm.SE3(0.01, 1.20, 1.0))

s19 = rp.Shape.Box(
    scale=[0.02, 0.62, 0.76],
    base=sm.SE3(0.03, 1.60, 0.38))

s20 = rp.Shape.Box(
    scale=[0.02, 0.64, 0.76],
    base=sm.SE3(0.60, 1.60, 0.38))

s21 = rp.Shape.Box(
    scale=[0.08, 0.60, 0.02],
    base=sm.SE3(0.04, 1.60, 0.75))

s22 = rp.Shape.Box(
    scale=[0.03, 0.03, 0.06],
    base=sm.SE3(0.04, 1.70, 0.78))

s23 = rp.Shape.Box(
    scale=[0.03, 0.03, 0.06],
    base=sm.SE3(0.04, 1.50, 0.78))

s24 = rp.Shape.Box(
    scale=[0.03, 0.03, 0.30],
    base=sm.SE3(0.04, 1.60, 0.90))

s25 = rp.Shape.Box(
    scale=[0.25, 0.03, 0.03],
    base=sm.SE3(0.15, 1.60, 1.05))

s26 = rp.Shape.Box(
    scale=[0.02, 0.02, 0.05],
    base=sm.SE3(0.265, 1.60, 1.025))

s27 = rp.Shape.Box(
    scale=[0.40, 0.02, 0.70],
    base=sm.SE3(0.20, 0.74, 1.55))

s28 = rp.Shape.Box(
    scale=[0.40, 0.02, 0.70],
    base=sm.SE3(0.20, 2.39, 1.55))

s29 = rp.Shape.Box(
    scale=[0.40, 0.02, 0.70],
    base=sm.SE3(0.20, 1.62, 1.55))

s30 = rp.Shape.Box(
    scale=[0.40, 1.67, 0.02],
    base=sm.SE3(0.20, 1.565, 1.20))

s31 = rp.Shape.Box(
    scale=[0.40, 1.67, 0.02],
    base=sm.SE3(0.20, 1.565, 1.90))

s32 = rp.Shape.Box(
    scale=[0.40, 1.67, 0.02],
    base=sm.SE3(0.20, 1.565, 1.55))

s33 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.76],
    base=sm.SE3(0.30, 1.29, 0.38))

s34 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.76],
    base=sm.SE3(0.30, 0.72, 0.38))

s35 = rp.Shape.Box(
    scale=[0.02, 0.6, 0.75],
    base=sm.SE3(0.02, 1.00, 0.37))

s36 = rp.Shape.Box(
    scale=[0.60, 0.6, 0.02],
    base=sm.SE3(0.30, 1.00, 0.35))

s37 = rp.Shape.Box(
    scale=[0.60, 0.6, 0.02],
    base=sm.SE3(0.30, 1.00, 0.01))

s38 = rp.Shape.Box(
    scale=[0.04, 0.04, 0.24],
    base=sm.SE3(0.43, 1.92, 0.87))

s39 = rp.Shape.Box(
    scale=[0.05, 0.05, 0.16],
    base=sm.SE3(0.45, 1.97, 0.81))

s40 = rp.Shape.Box(
    scale=[0.06, 0.06, 0.32],
    base=sm.SE3(0.36, 1.94, 0.92))

s41 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.75],
    base=sm.SE3(0.30, 2.39, 0.375))

s42 = rp.Shape.Box(
    scale=[0.60, 0.02, 0.75],
    base=sm.SE3(0.30, 1.91, 0.375))

s43 = rp.Shape.Box(
    scale=[0.02, 0.48, 0.16],
    base=sm.SE3(0.60, 2.16, 0.08))

s44 = rp.Shape.Box(
    scale=[0.02, 0.48, 0.08],
    base=sm.SE3(0.61, 2.16, 0.72))

s45 = rp.Shape.Box(
    scale=[0.02, 0.48, 0.74],
    base=sm.SE3(0.02, 2.15, 0.35))

s46 = rp.Shape.Box(
    scale=[0.60, 0.48, 0.02],
    base=sm.SE3(0.90, 2.16, 0.16))

s47 = rp.Shape.Box(
    scale=[0.56, 0.48, 0.02],
    base=sm.SE3(0.30, 2.15, 0.10))

s48 = rp.Shape.Box(
    scale=[0.03, 0.03, 0.07],
    base=sm.SE3(1.10, 2.00, 0.12))

s49 = rp.Shape.Box(
    scale=[0.03, 0.03, 0.07],
    base=sm.SE3(1.10, 2.3, 0.12))

s50 = rp.Shape.Box(
    scale=[0.03, 0.33, 0.03],
    base=sm.SE3(1.10, 2.15, 0.08))

s51 = rp.Shape.Box(
    scale=[0.2, 0.03, 0.26],
    base=sm.SE3(0.38, 2.34, 0.89))

s52 = rp.Shape.Box(
    scale=[0.2, 0.02, 0.3],
    base=sm.SE3(0.41, 2.37, 0.92))

s0 = rp.Shape.Sphere(
    radius=0.05
)

env = rp.backend.Swift()
env.launch()

r = rp.models.PR2()
r.q = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16825, 0.0,
    0.0, 0.0, -0.8451866735633127, 0.08128204585620136, -1.8311522093787793,
    -1.5550420276338315, 2.6911049983477024, -0.8325789594278508,
    2.067980015738538, 2.248201624865942e-15, 0.0, 0.0, 0.0, 0.0, 0.0,
    -1.1102230246251565e-16, 1.5301977050596074, -0.1606585554274158, 1.122,
    -2.1212501013292435, 2.5303972717977263, -0.7028113314291433,
    1.925250846169634, -7.494005416219807e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0]

R = sm.base.q2r([0.03074371707199644, -0.0, -0.0, -0.9995273002077517])
T = np.eye(4)
T[:3, :3] = R
T = sm.SE3(T)
r.base = sm.SE3(1.0746809244155884, 1.500715732574463, 0.0) * T

env.add(r)
# env.add(r, show_robot=False, show_collision=True)
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
env.add(s18)
env.add(s19)
env.add(s20)
env.add(s21)
env.add(s22)
env.add(s23)
env.add(s24)
env.add(s25)
env.add(s26)
env.add(s27)
env.add(s28)
env.add(s29)
env.add(s30)
env.add(s31)
env.add(s32)
env.add(s33)
env.add(s34)
env.add(s35)
env.add(s36)
env.add(s37)
env.add(s38)
env.add(s39)
env.add(s40)
env.add(s41)
env.add(s42)
env.add(s43)
env.add(s44)
env.add(s45)
env.add(s46)
env.add(s47)
env.add(s48)
env.add(s49)
env.add(s50)
env.add(s51)
env.add(s52)
time.sleep(3)

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

        # closest = 1000000
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

probs = 36
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
