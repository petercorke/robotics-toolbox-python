#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

# env = swift.Swift()
# env.launch()

panda = rp.models.Panda()
panda.q = panda.qr

panda = rp.FastRobot(panda)
print(panda)

print(panda.jacob0(panda.q))
print(panda.jacobe(panda.q))
