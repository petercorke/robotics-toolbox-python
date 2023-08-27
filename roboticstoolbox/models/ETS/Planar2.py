#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ET import ET2
from roboticstoolbox.robot.ETS import ETS2
from roboticstoolbox.robot.Robot import Robot2
from roboticstoolbox.robot.Link import Link2


class Planar2(Robot2):
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

        a1 = 1
        a2 = 1

        l0 = Link2(ETS2(ET2.R()), name="link0")
        l1 = Link2(ETS2(ET2.tx(a1)) * ET2.R(), name="link1", parent=l0)
        l2 = Link2(ETS2(ET2.tx(a2)), name="ee", parent=l1)

        super().__init__([l0, l1, l2], name="Planar2", comment="Planar 2D manipulator")

        self.qb = np.array([0, np.pi/2])
        self.qz = np.zeros(2)

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qb", self.qb)


if __name__ == "__main__":  # pragma nocover

    robot = Planar2()
    print(robot)

    # print(robot.fkine(robot.qz))
