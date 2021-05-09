# a circle of Puma robot's doing a Mexican wave

import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
from roboticstoolbox.backends.swift import Swift
import time

puma0 = rtb.models.URDF.Puma560()
swift = Swift()

swift.launch()

pumas = []

for theta in np.linspace(0, 2 * np.pi, 10):
    base = SE3.Rz(theta) * SE3(2, 0, 0)

    puma = rtb.ERobot(puma0)  # clone the robot
    puma.base = base
    puma.q = puma0.qz
    swift.add(puma)
    pumas.append(puma)

# the wave is a Gaussian that moves around the circle
tt = np.linspace(0, 10, 100)
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
g = gaussian(tt, 5, 1)

for t in range(1000):
    for i, puma in enumerate(pumas):
        k = (t + i * 10) % len(tt)
        puma.q = np.r_[0, g[k], -g[k], 0, 0, 0]

        # NEXT LINE CAUSES 9/10 ROBOTS TO DISAPPEAR 
        swift.step(0.1)
        time.sleep(0.02)
    