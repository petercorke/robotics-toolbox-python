#!/usr/bin/env python

from roboticstoolbox.robot.ERobot import ERobot
import numpy as np


class PR2(ERobot):
    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "pr2_description/robots/pr2.urdf.xacro", "pr2_description"
        )

        super().__init__(
            links,
            gripper_links=[links[53], links[73]],
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.grippers[0].tool = self.link_dict["r_gripper_tool_frame"].A()
        self.grippers[1].tool = self.link_dict["l_gripper_tool_frame"].A()

        self.manufacturer = "Willow Garage"

        self.qz = np.zeros(31)


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
