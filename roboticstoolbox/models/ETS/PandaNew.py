#!/usr/bin/env python

import numpy as np
from numpy.linalg import qr
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.ELink import ELink

from roboticstoolbox.robot.ET import ET, BaseET
import sympy
import cProfile
import roboticstoolbox as rtb
import spatialmath as sm
from copy import copy, deepcopy

import fknm


class Panda(ERobot):
    """
    Create model of Franka-Emika Panda manipulator

    panda = Panda() creates a robot object representing the Franka-Emika
    Panda robot arm. This robot is represented using the elementary
    transform sequence (ETS).

    ETS taken from [1] based on
    https://frankaemika.github.io/docs/control_parameters.html

    :references:
        - Kinematic Derivatives using the Elementary Transform
          Sequence, J. Haviland and P. Corke

    """

    def __init__(self):

        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = ELink(ETS.tz(0.333) * ETS.rz(), name="link0", parent=None)

        l1 = ELink(ETS.rx(-90 * deg) * ETS.rz(), name="link1", parent=l0)

        l2 = ELink(ETS.rx(90 * deg) * ETS.tz(0.316) * ETS.rz(), name="link2", parent=l1)

        l3 = ELink(
            ETS.tx(0.0825) * ETS.rx(90, "deg") * ETS.rz(), name="link3", parent=l2
        )

        l4 = ELink(
            ETS.tx(-0.0825) * ETS.rx(-90, "deg") * ETS.tz(0.384) * ETS.rz(),
            name="link4",
            parent=l3,
        )

        l5 = ELink(ETS.rx(90, "deg") * ETS.rz(), name="link5", parent=l4)

        l6 = ELink(
            ETS.tx(0.088) * ETS.rx(90, "deg") * ETS.tz(0.107) * ETS.rz(),
            name="link6",
            parent=l5,
        )

        ee = ELink(ETS.tz(tool_offset) * ETS.rz(-np.pi / 4), name="ee", parent=l6)

        elinks = [l0, l1, l2, l3, l4, l5, l6, ee]

        # super(Panda, self).__init__(
        #     elinks,
        #     name='Panda',
        #     manufacturer='Franka Emika')

        # self.addconfiguration(
        #     "qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        # self.addconfiguration(
        #     "qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]))


if __name__ == "__main__":  # pragma nocover

    r = rtb.models.Panda()
    # q = r.qr
    # q = [0, -0.3, 0, -2.2, 0, 2, 0.78539816]

    # # ET
    # x = sympy.symbols("x")
    # e1 = ET.Rx()
    # e2 = ET.Ry(0.7)
    # print(e1.T(0.7))
    # print(e2.T())

    # Variables
    base = np.eye(4)
    tool = np.eye(4)

    # BaseET("Rx", eta=5.0)

    # e = ET.

    # # Fkine
    # print(ets.fkine(q, base, tool))
    # print(ets.fkine(q, None, None))
    # print(e1.T(1.0) @ e2.T())

    # A Panda ETS
    ets = (
        ET.tz(0.333)
        * ET.Rz(jindex=0)
        * ET.Rx(-np.pi / 2)
        * ET.Rz(jindex=1)
        * ET.ty(-0.316)
        * ET.Rx(np.pi / 2)
        * ET.Rz(jindex=2)
        * ET.tx(0.0825)
        * ET.Rx(np.pi / 2)
        * ET.Rz(jindex=3)
        * ET.tx(-0.0825)
        * ET.ty(0.384)
        * ET.Rx(-np.pi / 2)
        * ET.Rz(jindex=4)
        * ET.Rx(np.pi / 2)
        * ET.Rz(jindex=5)
        * ET.tx(0.088)
        * ET.Rx(np.pi / 2)
        * ET.Rz(jindex=6)
        * ET.tz(0.107)
        * ET.tz(0.1034)
    )

    x = sympy.Symbol("x")
    q1 = np.r_[x, r.qr[1:]]
    q2 = r.qr
    print(q1.dtype)
    print(ets.jacob0(q1)[0, 0])
    print(ets.jacob0(q2)[0, 0])
    # print(q)

    r1 = rtb.ET.Rx(2.5)
    r4 = rtb.ET.Rx(2.5)
    r2 = copy(r1)
    r3 = deepcopy(r1)

    ets = r1 * r3
    print(r1 == r4)
    # print(r1.fknm)

    # print(r1.T(1.0))
    # print(r2.T(1.0))
    # print(r3.T(1.0))
    # print(r1._BaseET__fknm)
    # print(r2._BaseET__fknm)
    # print(r3._BaseET__fknm)

    # print(repr(ET.Rz(jindex=5)))

    # Jacob 0
    ql = [12.1, -45.3, 0, -2.2, 0, 2, 0.78539816]
    q = r.qr
    # print(q.__array_interface__)

    # print(np.round(r.jacob0(q), 2))
    # print(np.round(ets.jacob0(q), 2))

    tool = sm.SE3([sm.SE3(19.0, 0, 0), sm.SE3(17.0, 0, 0)])

    # tool = sm.SE3(19.0, 0, 0)
    # tool = np.eye(4)

    # print(tool[0].A)

    # # tool2 = sm.SE3(19.0, 0, 0)
    # print(type(tool.data[0]))
    # print(tool[0] is tool[0])

    # tool[0].data[0].__array_interface__

    # print(tool[0].__array_interface__)

    # print(tool.__array_interface__)
    # print(tool.__array_interface__ is tool.A.__array_interface__)
    # # tool.A[:] = tool2.A.copy()
    # # print(np.round(ets.jacob0(q, tool=np.eye(4)), 2)[0, 0:3])
    # print(np.round(ets.jacob0(q, tool=tool[0]), 2)[0, 0:3])

    # print(tool.A)

    # for i in range(1000000000):
    #     ets.jacob0(q)

    # print(np.round(ets.jacob0(q), 2))
    # print(type(q))
    # print()
    # q = np.array([12.1, 45.3, 0, -2.2, 0, 2, 0.78539816])

    # print(np.round(ets.jacob0(q), 2)[0, 0:3])
    # print(np.round(ets.jacob0(q), 2)[0, 0:3])
    # print(np.round(ets.jacob0(q), 2)[0, 0:3])
    # print(np.round(ets.jacob0(q), 2)[0, 0:3])
    # print()
    # q = np.array([[0, -0.3, 0, -2.2, 0, 2, 0.78539816]]).T
    # print(np.round(ets.jacob0(q), 2))

    # ets.sfuhsf = 28308
    # print(ets.__dict__)
    # print(ets.__slots__)

    # Profiling
    num = 1000

    # # Profile fkine
    # def new():
    #     for _ in range(num):
    #         ets.fkine(q)

    # def old():
    #     for _ in range(num):
    #         r.fkine(q, fast=True)

    # Profile jacob0
    def new():
        for _ in range(num):
            ets.jacob0(q)

    def old():
        for _ in range(num):
            r.jacob0(q)

    # cProfile.run("old()")
    # cProfile.run("new()")
