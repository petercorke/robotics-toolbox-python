#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

env.set_camera_pose([1.4, 0, 0.7], [0, 0.0, 0.5])

r = rtb.models.YuMi()
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


l_frame = sg.Axes(0.1, pose=r.grippers[0].tool)
r_frame = sg.Axes(0.1, pose=r.grippers[1].tool)

l_frame.attach_to(r.grippers[0].links[0])
r_frame.attach_to(r.grippers[1].links[0])

env.add(l_frame)
env.add(r_frame)


# Construct an ETS for the left and right arms
la = r.ets(end=r.grippers[0])
ra = r.ets(end=r.grippers[1])

arrivedl = False
arrivedr = False

dt = 0.05

gain = np.array([1, 1, 1, 1.6, 1.6, 1.6])

while not arrivedl or not arrivedr:

    vl, arrivedl = rtb.p_servo(la.fkine(r.q), lTep, gain=gain, threshold=0.001)
    vr, arrivedr = rtb.p_servo(ra.fkine(r.q), rTep, gain=gain, threshold=0.001)

    r.qd[la.jindices] = np.linalg.pinv(la.jacobe(r.q)) @ vl
    r.qd[ra.jindices] = np.linalg.pinv(ra.jacobe(r.q)) @ vr

    env.step(dt)

env.hold()
