# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

from roboticstoolbox.backends import Swift
from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import numpy as np
import pathlib
import os
import time

import cProfile


path = os.path.realpath('.')

env = Swift.Swift()
env.launch()

path = rtb.path_to_datafile('data')

r = rtb.models.Panda()
r.q = r.qr
r.qd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


g1 = rtb.Box(
    base=SE3(-1, 0, 0.5),
    scale=[0.1, 0.2, 0.3],
    color=[0.9, 0.9, 0.9, 1]
)
g1.v = [0.01, 0.0, 0, 0.1, 0, 0]

g2 = rtb.Sphere(
    base=SE3(),
    radius=0.1,
    color=[0.9, 0.9, 0.9, 1]
)
g2.v = [0.01, 0, 0.0, 0.0, 0, 0]


# env.add(g1)
# env.add(g2)
env.add(r, show_robot=False, show_collision=True)


def stepper():
    for i in range(1000):
        env.step(0.01)


cProfile.run('stepper()')

# for i in range(1000):
#     start = time.time()
#     env.step(0.01)
#     print(time.time() - start)

env.hold()
