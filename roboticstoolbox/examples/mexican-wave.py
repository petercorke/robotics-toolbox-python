#!/usr/bin/env python
"""
@author Peter Corke
@author Jesse Haviland
"""

# A circle of Puma robot's doing a Mexican wave

import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
from roboticstoolbox.backends.swift import Swift
import time

swift = Swift()
swift.launch()

puma0 = rtb.models.URDF.Puma560()
pumas = []
num_robots = 15
rotation = 2 * np.pi * ((num_robots - 1) / num_robots)

for theta in np.linspace(0, rotation, num_robots):
    base = SE3.Rz(theta) * SE3(2, 0, 0)

    # Clone the robot
    puma = rtb.ERobot(puma0)
    puma.base = base
    puma.q = puma0.qz
    swift.add(puma)
    pumas.append(puma)

# The wave is a Gaussian that moves around the circle
tt = np.linspace(0, num_robots, num_robots * 10)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


g = gaussian(tt, 5, 1)
t = 0

while True:
    for i, puma in enumerate(pumas):
        k = (t + i * 10) % len(tt)
        puma.q = np.r_[0, g[k], -g[k], 0, 0, 0]

        swift.step(0)
        time.sleep(0.001)

    t += 1
