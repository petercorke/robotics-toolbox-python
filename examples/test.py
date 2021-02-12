# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import pathlib
import os

path = os.path.realpath('.')

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

b1 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 0.20))

b2 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 0.60))

b3 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 1.00))

b4 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 1.40))

b5 = rtb.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(1.95, 0.55, 0.7))

b6 = rtb.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(1.95, -0.55, 0.7))

collisions = [b1, b2, b3, b4, b5, b6]

path = pathlib.Path(path) / 'roboticstoolbox' / 'data'

g1 = rtb.Mesh(
    filename=str(path / 'gimbal-ring1.stl'),
    base=sm.SE3(0, 0, 2.0)
)
g1.v = [0, 0, 0, 0.4, 0, 0]

g2 = rtb.Mesh(
    filename=str(path / 'gimbal-ring2.stl'),
    base=sm.SE3(0, 0, 2.0)
)
g2.v = [0, 0, 0, 0, 0.4, 0]

g3 = rtb.Mesh(
    filename=str(path / 'gimbal-ring3.stl'),
    base=sm.SE3(0, 0, 2.0)
)
g3.v = [0, 0, 0, 0, 0, 0.4]

env.add(g1)
env.add(g2)
env.add(g3)

while(True):
    env.step(0.05)
