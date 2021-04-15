#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import cProfile
# import numpy.testing as nt

env = swift.Swift(_dev=False)
env.launch()


panda = rtb.models.Panda()
panda.q = panda.qr
env.add(panda, show_robot=True)


ev = [0.01, 0, 0, 0, 0, 0]
panda.qd = np.linalg.pinv(panda.jacobe(panda.q, fast=True)) @ ev
env.step(0.001)


def stepper():
    for i in range(10000):
        panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
        # panda.fkine_all_fast(panda.q)
        env.step(0.001)


cProfile.run('stepper()')
