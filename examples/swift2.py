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

from roboticstoolbox.backends import swift
from math import pi
import roboticstoolbox as rtb
from spatialmath import SO3, SE3
import spatialmath as sm
import numpy as np
import pathlib
import os
import time

import cProfile


from spatialmath.base import r2q


pm = rtb.models.DH.Panda()
p = rtb.models.ETS.Panda()
p2 = rtb.models.Panda()
q = np.array([1, 2, 3, 4, 5, 6, 7])
p.q = q
pm.q = q

p.fkine_all(q)
p2.fkine_all(q)
r2 = pm.fkine_all(q)

for i in range(7):
    print(np.round(p.links[i]._fk, 2))
    # print(np.round(p2.links[i]._fk, 2))
    print(np.round(r2[i].A, 2))

    print()
    print()

# import swift


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

env = swift.Swift(_dev=True)
env.launch()


def slidercb(e):
    print(e)


def selectcb(e):
    print(e)


def checkcb(e):
    print(e)
    # select.value = e


label = swift.Label('Demo')
slider = swift.Slider(slidercb, 10, 95, 5, 15, 'slider this is', ' d')
select = swift.Select(selectcb, 'selec', [
                      'on', 'tw', 'three'], 2)

check = swift.Checkbox(checkcb, 'checkbox', [
    'on', 'tw', 'three'], [False, True, True])


def buttoncb(e):
    print('BUTTON')
    # check.checked = [True, True, False]
    # check.desc = 'new desc'
    slider.value = 60


button = swift.Button(buttoncb, 'button')


def radiocb(e):
    print(e)
    select.value = e


radio = swift.Radio(radiocb, 'radio', [
    'on', 'tw', 'three'], 2)

# env.add(label)
# env.add(slider)
# env.add(button)
# env.add(select)
# env.add(radio)
# env.add(check)

# while True:
#     env.step(0.05)
#     time.sleep(0.001)

# env.hold()


panda = rtb.models.Panda()
panda.q = panda.qr
# panda.base = sm.SE3(1.2, 1, 0) * sm.SE3.Rz(np.pi/2)
env.add(panda, show_robot=True)


ev = [0.01, 0, 0, 0, 0, 0]
panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
env.step(0.001)


def stepper():
    for i in range(10000):
        panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
        # panda.fkine_all_fast(panda.q)
        env.step(0.001)


# box = rtb.Box([0.8, 0.1, 0.1])
# env.add(box)

# stepper()
# env.remove(panda)


# r = rtb.models.LBR()
# r.q = r.qr
# r.qd = [0.01, 0.01, 0.01, 0, 0, 0, 0]
# env.add(r)
# for i in range(10000):
#     panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
#     env.step(0.001)

# # env.remove(box)
cProfile.run('stepper()')

# env._run_thread = False
# env.restart()

# r = rtb.models.LBR()
# r.q = r.qr
# r.qd = [0.01, 0.01, 0.01, 0, 0, 0, 0]
# env.add(r)
# for i in range(1000000):
#     panda.qd = np.linalg.pinv(panda.jacob0(panda.q, fast=True)) @ ev
#     env.step(0.001)

# env.reset()
# env.add(r)
# env.close()

# cProfile.run('stepper()')

# print(panda.fkine(panda.q))
# print(panda.fkine_fast(panda.q))

env.hold()
