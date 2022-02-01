#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.ELink import ELink
from roboticstoolbox.robot.ET import ET
import roboticstoolbox as rtb

from spatialmath import SE3
import numpy.testing as nt


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

        l0 = ELink(ET.tz(0.333) * ET.Rz(), name="link0", parent=None)

        l1 = ELink(ET.Rx(-90 * deg) * ET.Rz(), name="link1", parent=l0)

        l2 = ELink(ET.Rx(90 * deg) * ET.tz(0.316) * ET.Rz(), name="link2", parent=l1)

        l3 = ELink(ET.tx(0.0825) * ET.Rx(90 * deg) * ET.Rz(), name="link3", parent=l2)

        l4 = ELink(
            ET.tx(-0.0825) * ET.Rx(-90 * deg) * ET.tz(0.384) * ET.Rz(),
            name="link4",
            parent=l3,
        )

        l5 = ELink(ET.Rx(90 * deg) * ET.Rz(), name="link5", parent=l4)

        l6 = ELink(
            ET.tx(0.088) * ET.Rx(90 * deg) * ET.tz(0.107) * ET.Rz(),
            name="link6",
            parent=l5,
        )

        ee = ELink(ET.tz(tool_offset) * ET.Rz(-np.pi / 4), name="ee", parent=l6)

        elinks = [l0, l1, l2, l3, l4, l5, l6, ee]

        super(Panda, self).__init__(elinks, name="Panda", manufacturer="Franka Emika")

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.logconfiguration("qr", self.qr)
        self.logconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    r = rtb.models.Panda()
    r2 = rtb.models.ETS.Panda()

    print(r2)

    # r2.addconfiguration("ready", [1, 2, 3, 4, 4], "deg")
    # qq = r2.configs["my_q"]

    # # r.ets()

    # for link in r:
    #     print(link.name)

    # for link in r.grippers[0].links:
    #     print(link.name)

    # print()

    # print(r.ets(start="panda_hand", end=r.links[0]))

    # a = r.qz

    # deg = np.pi / 180
    # mm = 1e-3
    # tool_offset = (103) * mm

    # q1 = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    # q2 = np.array([[0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]])
    # q3 = np.array([[0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]]).T
    # q4 = [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]
    # q5 = np.r_[q2, q2, q2, q2]

    # man = (
    #     SE3.Tz(0.333)
    #     * SE3.Rz(q1[0])
    #     * SE3.Rx(-90 * deg)
    #     * SE3.Rz(q1[1])
    #     * SE3.Rx(90 * deg)
    #     * SE3.Tz(0.316)
    #     * SE3.Rz(q1[2])
    #     * SE3.Tx(0.0825)
    #     * SE3.Rx(90 * deg)
    #     * SE3.Rz(q1[3])
    #     * SE3.Tx(-0.0825)
    #     * SE3.Rx(-90 * deg)
    #     * SE3.Tz(0.384)
    #     * SE3.Rz(q1[4])
    #     * SE3.Rx(90 * deg)
    #     * SE3.Rz(q1[5])
    #     * SE3.Tx(0.088)
    #     * SE3.Rx(90 * deg)
    #     * SE3.Tz(0.107)
    #     * SE3.Rz(q1[6])
    #     * SE3.Tz(tool_offset)
    #     * SE3.Rz(-np.pi / 4)
    # )

    # print(man)

    # # Normal q
    # nt.assert_almost_equal(man.A, r.fkine(q1))
    # nt.assert_almost_equal(man.A, r.fkine(q2))
    # nt.assert_almost_equal(man.A, r.fkine(q3))
    # nt.assert_almost_equal(man.A, r.fkine(q4))

    # # Traj q
    # fk_traj = r.fkine(q5)
    # for fk in fk_traj:
    #     nt.assert_almost_equal(man.A, fk)

    # # Tool
    # t1 = np.eye(4) * SE3.Ty(0.3).A
    # t2 = SE3(t1, check=False)
    # nt.assert_almost_equal(man.A @ t1, r.fkine(q1, tool=t1))
    # nt.assert_almost_equal((man * t2).A, r.fkine(q1, tool=t2))

    # # Base
    # b1 = np.eye(4) * SE3.Tx(0.3).A
    # b2 = SE3(b1, check=False)
    # r.base = b1
    # nt.assert_almost_equal(b1 @ man.A, r.fkine(q1))
    # r.base = b2
    # nt.assert_almost_equal((b2 * man).A, r.fkine(q1))

    # # Base and tool
    # r.base = b2
    # nt.assert_almost_equal((b2 * man * t2).A, r.fkine(q1, tool=t2))

    # # Don't include the base
    # r.base = b2
    # nt.assert_almost_equal((man * t2).A, r.fkine(q1, tool=t2, include_base=False))
