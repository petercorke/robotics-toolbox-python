# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

from roboticstoolbox.backends import Swift
from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import spatialmath as sm
import numpy as np
import pathlib
import os
import time

import cProfile

from spatialmath.base.sm_numba import numba_njit, use_numba, using_numba
import numba


from spatialmath.base import r2q


# num = 500000
# b = np.random.randn(num)
# sm.base.trotz(1.0)

# def stepper():
#     for i in range(num):
#         sm.base.trotz(b[i])


# cProfile.run('stepper()')

# print(using_numba())
# use_numba(False)
# print(using_numba())

# path = os.path.realpath('.')

env = Swift.Swift()
env.launch()

path = rtb.path_to_datafile('data')

r = rtb.models.Panda()
r.q = r.qr
# r.qd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# g1 = rtb.Box(
#     base=SE3(-1, 0, 0.5),
#     scale=[0.1, 0.2, 0.3],
#     color=[0.9, 0.9, 0.9, 1]
# )
# g1.v = [0.01, 0.0, 0, 0.1, 0, 0]

# g2 = rtb.Sphere(
#     base=SE3(),
#     radius=0.1,
#     color=[0.9, 0.9, 0.9, 1]
# )
# g2.v = [0.01, 0, 0.0, 0.0, 0, 0]


# # env.add(g1)
# # env.add(g2)
env.add(r, show_robot=False, show_collision=True)

ev = [0.1, 0, 0, 0, 0, 0]
r.jacob0(r.q)


def stepper():
    for i in range(1000):
        r.qd = np.linalg.pinv(r.jacob0(r.q)) @ ev
        env.step(0.004)
        # r.jacob0(r.q)


env.step(0.01)

cProfile.run('stepper()')

# for i in range(1000):
#     start = time.time()
#     env.step(0.01)
#     print(time.time() - start)

# env.hold()
