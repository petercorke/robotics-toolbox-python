#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class PR2(URDFRobot):
    """
    Class that imports a PR2 URDF model

    ``PR2()`` is a class which imports a Willow Garage PR2 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.PR2()
        >>> print(robot)

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        super().__init__("pr2", manufacturer="Willow Garage")

        self.qr = np.zeros(self.n)
        self.qz = np.zeros(self.n)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.qdlim = 2.0 * np.ones(self.n)


if __name__ == "__main__":  # pragma nocover
    r = PR2()

    # i = 0

    # for link in r.links:
    #     if link.isjoint:
    #         print(i, link.name)

    #         i += 1

    # path, n, t = r.get_path(end=r.grippers[0])

    # print(n)
    # print(t)

    # for l in path[1:]:
    #     if len(l.collision) > 0:
    #         print(l.isjoint)
    #         print(l.name)
    #         print(l.parent.name)
    #         print()

    # for l in r.grippers[0].links:
    #     if len(l.collision) > 0:
    #         print(l.isjoint)
    #         print(l.name)
    #         print(l.parent.name)
    #         print()
