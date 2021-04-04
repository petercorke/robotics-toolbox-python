#!/usr/bin/env python

from roboticstoolbox.robot.ERobot import ERobot


class PR2(ERobot):

    def __init__(self):

        links, name = self.URDF_read(
            "pr2_description/robots/pr2.urdf.xacro",
            "pr2_description")

        super().__init__(
            links,
            name=name)

        self.manufacturer = 'Willow Garage'

if __name__ == '__main__':   # pragma nocover

    robot = PR2()
    print(robot)