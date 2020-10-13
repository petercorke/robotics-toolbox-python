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
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 4.5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')

fig, ax = plt.subplots()
fig.set_size_inches(2.5, 1.5)

ax.set(xlabel='Time (s)', ylabel='Distance (m)')
ax.grid()
plt.grid(True)
ax.set_xlim(xmin=0, xmax=12.1)
ax.set_ylim(ymin=0, ymax=0.7)
plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)


pld0 = ax.plot(
    [0], [0], label='Distance to Obstacle 1')

pld1 = ax.plot(
    [0], [0], label='Distance to Obstacle 2')

pld2 = ax.plot(
    [0], [0], label='Distance to Goal')

plm = ax.plot(
    [0], [0], label='Manipuability')

ax.legend()
ax.legend(loc="lower right")

plt.ion()
plt.show()

qdmax = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
    5000000, 5000000, 5000000, 5000000, 5000000, 5000000])
lb = -qdmax
ub = qdmax

q0 = [-0.5653, -0.1941, -1.2602, -0.7896, -2.3227, -0.3919, -2.5173]

s0 = rp.Shape.Sphere(
    radius=0.05,
    base=sm.SE3(0.45, 0.4, 0.3)
)

s1 = rp.Shape.Sphere(
    radius=0.05,
    base=sm.SE3(0.1, 0.35, 0.65)
)

s2 = rp.Shape.Sphere(
    radius=0.02,
    base=sm.SE3(0.3, 0, 0)
)

s0.v = [0.01, -0.2, 0, 0, 0, 0]
# s1.v = [0, -0.2, 0, 0, 0, 0]
# s2.v = [0, 0.1, 0, 0, 0, 0]

env = rp.backend.Swift()
env.launch()

r = rp.models.Panda()

n = 7
env.add(r)
env.add(s0)
# env.add(s1)
env.add(s2)
time.sleep(1)


s0.x = []
s1.x = []
s2.x = []
s0.y = []
s1.y = []
s2.y = []
m = []

def update_graph2(ob, graph):
    # Update the robot links

    graph[0].set_xdata(ob.x)
    graph[0].set_ydata(ob.y)


def link_calc(link, col, ob, q):
    dii = 5
    di = 0.3
    ds = 0.05

    ret = p.getClosestPoints(col.co, ob.co, dii)

    if len(ret) > 0:
        ret = ret[0]
        wTlp = sm.SE3(ret[5])
        wTcp = sm.SE3(ret[6])
        lpTcp = wTlp.inv() * wTcp

        d = ret[8]

    if d < di:
        n = lpTcp.t / d
        nh = np.expand_dims(np.r_[n, 0, 0, 0], axis=0)

        Je = r.jacobe(q, from_link=r.base_link, to_link=link, offset=col.base)
        n = Je.shape[1]
        dp = nh @ ob.v
        l_Ain = np.zeros((1, 13))
        l_Ain[0, :n] = nh @ Je
        l_bin = (0.8 * (d - ds) / (di - ds)) + dp
    else:
        l_Ain = None
        l_bin = None

    return l_Ain, l_bin, d, wTcp


def servo(q0, Tep, it):

    r.q = q0
    r.fkine_all()
    r.qd = np.zeros(r.n)
    env.step(1)
    links = r._fkpath[1:]

    arrived = False
    i = 0
    Q = 0.1 * np.eye(n + 6)

    while not arrived and i < it:

        if i > (4 / 0.05):
            s2.v = [0, 0, 0, 0, 0, 0]

        Tep.A[:3, 3] = s2.base.t
        Tep.A[2, 3] += 0.1
        q = r.q
        v, arrived = rp.p_servo(r.fkine(), Tep, 0.5, 0.05)

        eTep = r.fkine().inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

        Q[n:, n:] = 1 * (1 / e) * np.eye(6)
        Aeq = np.c_[r.jacobe(), np.eye(6)]
        beq = v.reshape((6,))
        Jm = r.jacobm().reshape(7,)
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
                obj = s0
                l_Ain, l_bin, ret, wTcp = link_calc(link, col, obj, q[:j])
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

        s0.y.append(closest)
        s0.x.append(i * 0.05)

        s2.y.append(np.linalg.norm(eTep.t))
        s2.x.append(i * 0.05)
        m.append(r.manipulability())


        try:
            qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        except (ValueError, TypeError):
            print("Value Error")
            break

        r.qd = qd[:7]

        i += 1
        env.step(50)

        # r.loc = np.c_[r.loc, r.fkine().t]
        # s0.loc = np.c_[s0.loc, s0.base.t]
        # s1.loc = np.c_[s1.loc, s1.base.t]
        # s2.loc = np.c_[s2.loc, s2.base.t]

        # update_graph2(r, plr)
        update_graph2(s0, pld0)
        update_graph2(s2, pld2)
        plm[0].set_xdata(s2.x)
        plm[0].set_ydata(m)
        plt.pause(0.001)

    return arrived


q0 = r.qr
r.q = q0

s2.base = sm.SE3.Tx(0.6) * sm.SE3.Tz(0.1) * sm.SE3.Ty(-0.2) * sm.SE3.Tz(-0.1)

Tep = r.fkine()
Tep.A[:3, 3] = s2.base.t


servo(q0, Tep, 5000)

obj = {
    's0': [s0.x, s0.y],
    's1': [s1.x, s1.y],
    's2': [s2.x, s2.y],
    'm': [s0.x, m]
}

pickle.dump(obj, open('neo1.p', 'wb'))

plt.ioff()
plt.show()
