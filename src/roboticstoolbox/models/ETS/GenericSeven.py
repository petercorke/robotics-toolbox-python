#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Link import Link


class GenericSeven(Robot):
    """
    Create model of a generic seven degree-of-freedom robot

    robot = GenericSeven() creates a robot object. This robot is represented
    using the elementary transform sequence (ETS).

    """

    def __init__(self):

        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        link_length = 0.5

        l0 = Link(ET.tz(link_length) * ET.Rz(), name="link0", parent=None)

        l1 = Link(ETS(ET.Ry()), name="link1", parent=l0)

        l2 = Link(ET.tz(link_length) * ET.Rz(), name="link2", parent=l1)

        l3 = Link(ETS(ET.Ry()), name="link3", parent=l2)

        l4 = Link(ET.tz(link_length) * ET.Rz(), name="link4", parent=l3)

        l5 = Link(ETS(ET.Ry()), name="link5", parent=l4)

        l6 = Link(ET.tx(link_length) * ET.Rz(), name="link6", parent=l5)

        ee = Link(ETS(ET.tz(-link_length)), name="ee", parent=l6)

        elinks = [l0, l1, l2, l3, l4, l5, l6, ee]
        # elinks = [l0, l1, l2, l3, l4, l5, ee]

        super(GenericSeven, self).__init__(
            elinks, name="Generic Seven", manufacturer="Jesse's Imagination"
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
