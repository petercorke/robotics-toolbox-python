#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends import swift
import roboticstoolbox as rtb
import numpy as np
import cProfile

env = swift.Swift(_dev=False)
env.launch()


# panda = rtb.models.Panda()
# panda.q = panda.qr
# env.add(panda, show_robot=True)


# ev = [0.01, 0, 0, 0, 0, 0]
# panda.qd = np.linalg.pinv(panda.jacobe(panda.q, fast=True)) @ ev
# env.step(0.001)


# def stepper():
#     for i in range(10000):
#         panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
#         # panda.jacob0(panda.q, fast=True)

#         # panda.fkine_all_fast(panda.q)
#         env.step(0.001)


import spatialgeometry as sg
import spatialmath as sm

b1 = sg.Box(scale=[0.2, 0.1, 0.2], base=sm.SE3(1.0, 1, 1))
b2 = sg.Cylinder(radius=0.2, length=0.8, base=sm.SE3(0.0, 1, 1))

env.add(b1)
env.add(b2)

def stepper():
    for i in range(100000):
        a, b, c = b1.closest_point(b2, inf_dist=4.0, homogenous=False)

a, b, c = b1.closest_point(b2, inf_dist=4.0, homogenous=False)
cProfile.run("stepper()")
