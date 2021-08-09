#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import spatialgeometry as sg
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

r = rp.models.YuMi()
env.add(r)

lTep = (
    sm.SE3.Tx(0.45)
    * sm.SE3.Ty(0.25)
    * sm.SE3.Tz(0.3)
    * sm.SE3.Rx(np.pi)
    * sm.SE3.Rz(np.pi / 2)
)

rTep = (
    sm.SE3.Tx(0.45)
    * sm.SE3.Ty(-0.3)
    * sm.SE3.Tz(0.3)
    * sm.SE3.Rx(np.pi)
    * sm.SE3.Rz(np.pi / 2)
    * sm.SE3.Rx(np.pi / 5)
)

l_target = sg.Sphere(0.01, color=[0.2, 0.4, 0.65, 0.5], base=lTep)
l_target_frame = sg.Axes(0.1, base=lTep)

r_target = sg.Sphere(0.01, color=[0.64, 0.4, 0.2, 0.5], base=rTep)
r_target_frame = sg.Axes(0.1, base=rTep)

env.add(l_target)
env.add(l_target_frame)
env.add(r_target)
env.add(r_target_frame)


l_frame = sg.Axes(0.1)
r_frame = sg.Axes(0.1)
env.add(l_frame)
env.add(r_frame)

l_path, l_n, _ = r.get_path(end=r.grippers[0])
r_path, r_n, _ = r.get_path(end=r.grippers[1])

# Inner list comprehension gets a list jindicies from the links in l_path
# Outer list comprehension removes None's from the list (a None kindex means
# the link is static)
l_jindex = [i for i in [link.jindex for link in l_path] if i]
r_jindex = [i for i in [link.jindex for link in r_path] if i is not None]

arrivedl = False
arrivedr = False

dt = 0.05

gain = np.array([1, 1, 1, 1.6, 1.6, 1.6])

while not arrivedl or not arrivedr:

    vl, arrivedl = rp.p_servo(
        r.fkine(r.q, end=r.grippers[0]), lTep, gain=gain, threshold=0.001
    )
    vr, arrivedr = rp.p_servo(
        r.fkine(r.q, end=r.grippers[1]), rTep, gain=gain, threshold=0.001
    )

    r.qd[l_jindex] = np.linalg.pinv(r.jacob0(r.q, end=r.grippers[0])) @ vl
    r.qd[r_jindex] = np.linalg.pinv(r.jacob0(r.q, end=r.grippers[1])) @ vr

    l_frame.base = r.fkine(r.q, end=r.grippers[0])
    r_frame.base = r.fkine(r.q, end=r.grippers[1])

    env.step(dt)

env.hold()
