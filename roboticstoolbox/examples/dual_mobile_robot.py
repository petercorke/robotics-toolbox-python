#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import numpy as np
import time

env = swift.Swift()
env.launch(realtime=True)

omni = rtb.models.ETS.Omni()
omni.grippers[0].tool = (
    sm.SE3(-0.15, -0.1, 0.35) * sm.SE3.Rz(np.pi / 4) * sm.SE3.Rx(np.pi)
)
env.add(omni)


ax_frankie = sg.Axes(0.1)
ax_omni = sg.Axes(0.1)
env.add(ax_frankie)
env.add(ax_omni)


frankie = rtb.models.Frankie()
frankie.q = frankie.qr
frankie.base = sm.SE3(3, 0.1, 0) * sm.SE3.Rz(np.pi)
env.add(frankie)

time.sleep(2)

arrived = False

dt = 0.05

while not arrived:

    wTo = omni.fkine(omni.q, fast=True)
    wTf = frankie.fkine(frankie.q, fast=True)

    vo, arrivedo = rtb.p_servo(wTo, wTf, 1)
    vf, arrivedf = rtb.p_servo(wTf, wTo, 1)

    omni.qd = np.linalg.pinv(omni.jacobe(omni.q, fast=True)) @ vo
    frankie.qd = np.linalg.pinv(frankie.jacobe(frankie.q, fast=True)) @ vf

    ax_omni.base = wTo
    ax_frankie.base = wTf
    env.step(dt)

    # Reset bases
    omni.base = omni.fkine(omni.q, end=omni.links[2])
    omni.q = omni.qz
    base_new = frankie.fkine(frankie._q, end=frankie.links[2], fast=True)
    # base_new[2, 3] -= 0.38
    frankie._base.A[:] = base_new
    frankie.q[:2] = 0

env.hold()
