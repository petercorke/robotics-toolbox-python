#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.robot.ELink import ELink


class Planar_Y(ERobot):
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

        deg = np.pi/180
        mm = 1e-3
        tool_offset = 1

        # trunk of the tree
        l0 = ELink(
            ETS.rz(),
            name='link0',
            jindex=0,
            parent=None
        )

        l1 = ELink(
            ETS.tx(1) * ETS.rz(),
            name='link1',
            jindex=1,
            parent=l0
        )

        # branch 1
        l2 = ELink(
            ETS.tx(1) * ETS.rz(),
            name='link2a',
            jindex=2,
            parent=l1
        )

        l3 = ELink(
            ETS.tx(1) * ETS.rz(),
            name='link3a',
            jindex=3,
            parent=l2
        )

        eea = ELink(
            ETS.tz(tool_offset),
            name='eea',
            parent=l3
        )

        # branch 2
        l4 = ELink(
            ETS.tx(1) * ETS.rz(),
            name='link2b',
            jindex=4,
            parent=l1
        )

        l5 = ELink(
            ETS.tx(1) * ETS.rz(),
            name='link3b',
            jindex=5,
            parent=l4
        )

        eeb = ELink(
            ETS.tz(tool_offset),
            name='eeb',
            parent=l5
        )

        elinks = [l0, l1, l2, l3, l4, l5, eea, eeb]

        super().__init__(
            elinks,
            name='Planar-Y'
            )

        self.addconfiguration(
            "qz", [0, 0, 0, 0, 0, 0])
        self.addconfiguration(
            "qy", [0, 0, np.pi/4, 0, -np.pi/4, 0])


if __name__ == '__main__':   # pragma nocover

    robot = Planar_Y()
    print(robot)

    q = robot.qy
    print(robot.fkine(q, endlink='eea'))
    print(robot.fkine(q, endlink='eeb'))

    print(robot.fkine(q, endlink='eeb', startlink='eea'))

    # print(robot.n)
    # print(robot.elinks)
    # for link in robot:
    #     print(link.name, link)
    #     print(link.ets, link.v, link.isjoint)
    #     print(link.parent, link.child)
    #     print(link.Ts)
    #     print(link.geometry, link.collision)

    #     print()

    # from ansitable import ANSITable, Column

    # table = ANSITable(
    #     Column("link"),
    #     Column("parent"),
    #     Column("ETS", headalign="^", colalign="<"),
    #     border="thin")
    # for link in robot:
    #     if link.isjoint:
    #         color = ""
    #     else:
    #         color = "<<blue>>"
    #     table.row(
    #         color + link.name,
    #         link.parent.name if link.parent is not None else "-",
    #         link.ets())
    # table.print()
