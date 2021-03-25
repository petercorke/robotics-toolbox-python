#!/usr/bin/env python
"""
@author Jesse Haviland
"""

# import fknm
# import numpy as np
# import spatialmath as sm
# import cProfile
# import roboticstoolbox as rtb

# l = rtb.ELink(ets=rtb.ETS.rx(1.0), v=rtb.ETS.rz(flip=True))

# # print(l.isjoint)
# # print(l._v.isflip)
# # print(l._v.axis)
# # print(l._Ts.A)

# c = fknm.link_init(l.isjoint, l._v.isflip, 2, l._Ts.A)

# arr = np.empty((4, 4))
# fknm.link_A(1.0, c, arr)
# print(arr)
# print(l.A(1))

# # l._Ts.A[1, 1] = 99
# # l._isjoint = 2
# # fknm.link_update(c, l.isjoint, l._v.isflip, 2, l._Ts.A)

# # fknm.link_A(c, arr, 1.0)
# # print(arr)

# # print(l.A(1, True))


# # a = np.empty((4, 4))

# # fknm.rz(2, np.array([1.0, 2.0]))
# # fknm.rz(np.pi, a)
# # print(sm.base.trotz(np.pi))

# # print(a)
# # sm.base.trotz(1)


# def Rz(eta):
#     arr = np.empty((4, 4))
#     fknm.link_A(eta, c, arr)
#     return arr


# def cc(it):
#     for i in range(it):
#         # arr = np.empty((4, 4))
#         # fknm.rz(i, arr)
#         a = Rz(i)


# def slow(it):
#     for i in range(it):
#         a = l.A(i)


# it = 10000
# cProfile.run('cc(it)')
# cProfile.run('slow(it)')

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

# ur = rtb.models.UR5()
# ur.base = sm.SE3(0.3, 1, 0) * sm.SE3.Rz(np.pi/2)
# ur.q = [-0.4, -np.pi/2 - 0.3, np.pi/2 + 0.3, -np.pi/2, -np.pi/2, 0]
# env.add(ur)

# lbr = rtb.models.LBR()
# lbr.base = sm.SE3(1.8, 1, 0) * sm.SE3.Rz(np.pi/2)
# lbr.q = lbr.qr
# env.add(lbr)

# k = rtb.models.KinovaGen3()
# k.q = k.qr
# k.q[0] = np.pi + 0.15
# k.base = sm.SE3(0.7, 1, 0) * sm.SE3.Rz(np.pi/2)
# env.add(k)

env = Swift.Swift()
env.launch()


panda = rtb.models.Panda()
panda.q = panda.qr
# panda.base = sm.SE3(1.2, 1, 0) * sm.SE3.Rz(np.pi/2)
env.add(panda, show_robot=True)


ev = [0.01, 0, 0, 0, 0, 0]


def stepper():
    for i in range(1000):
        panda.qd = np.linalg.pinv(panda.jacob0_fast(panda.q)) @ ev
        env.step(0.004)


env.step(0.01)

# stepper()

cProfile.run('stepper()')

print(panda.fkine(panda.q))
print(panda.fkine_fast(panda.q))

# env.hold()
