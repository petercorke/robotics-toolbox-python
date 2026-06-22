#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot


class PR2(Robot):
    _urdf_path = "pr2_description/robots/pr2.urdf.xacro"
    _manufacturer = "Willow Garage"

    def __init__(self):

        super().__init__()

        self.grippers[0].tool = self.link_dict["r_gripper_tool_frame"].A()
        self.grippers[1].tool = self.link_dict["l_gripper_tool_frame"].A()

        self.qr = np.zeros(31)
        self.qz = np.zeros(31)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.qdlim = 2.0 * np.ones(31)


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
