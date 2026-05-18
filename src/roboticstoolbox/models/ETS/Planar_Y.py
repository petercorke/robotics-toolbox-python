#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Link import Link


class Planar_Y(Robot):
    """
    Create model of a branched planar manipulator::

        L0 -- L1 -+- L2a -- L3a -- EEa
                |
                +- L2b -- L3b -- EEb

    ``Planar_Y()`` creates a planar branched manipulator model.


    :references:
        - Kinematic Derivatives using the Elementary Transform
          Sequence, J. Haviland and P. Corke

    """

    def __init__(self):

        # deg = np.pi / 180
        # mm = 1e-3
        tool_offset = 1

        # trunk of the tree
        l0 = Link(ETS(ET.Rz()), name="link0", jindex=0, parent=None)

        l1 = Link(ET.tx(1) * ET.Rz(), name="link1", jindex=1, parent=l0)

        # branch 1
        l2 = Link(ET.tx(1) * ET.Rz(), name="link2a", jindex=2, parent=l1)

        l3 = Link(ET.tx(1) * ET.Rz(), name="link3a", jindex=3, parent=l2)

        eea = Link(ETS(ET.tz(tool_offset)), name="eea", parent=l3)

        # branch 2
        l4 = Link(ET.tx(1) * ET.Rz(), name="link2b", jindex=4, parent=l1)

        l5 = Link(ET.tx(1) * ET.Rz(), name="link3b", jindex=5, parent=l4)

        eeb = Link(ETS(ET.tz(tool_offset)), name="eeb", parent=l5)

        elinks = [l0, l1, l2, l3, l4, l5, eea, eeb]

        super().__init__(elinks, name="Planar-Y")

        self.qr = np.array([0, 0, np.pi / 4, 0, -np.pi / 4, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = Planar_Y()
    print(robot)
